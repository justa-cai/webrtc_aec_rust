//! 匹配滤波器组（时延估计核心），对照
//! `modules/audio_processing/aec3/matched_filter.{h,cc}`。
//!
//! 5 个 512 抽头（降采样域）NLMS 滤波器，彼此错开 384 抽头，覆盖约 0~608 ms。
//! 每个滤波器独立自适应；`h` 的能量峰位置即该滤波器覆盖范围内的回声延迟。
//! 峰值可靠（不在边界 + 残差显著小于不滤波基线）且残差最小的滤波器为 winner。
//!
//! 前回声（pre-echo）：对 winner 滤波器额外记录"部分卷积和"的误差，
//! 从峰值向更小延迟方向扫描，找出回声实质开始的最早位置。

use crate::constants::{
    ACCUMULATED_ERROR_SMOOTH_UP, ACCUMULATED_ERROR_SUB_SAMPLE_RATE,
    MATCHED_FILTER_PEAK_HIGH_MARGIN, MATCHED_FILTER_PEAK_LOW_MARGIN,
    MATCHED_FILTER_SATURATION_LIMIT, PRE_ECHO_MIN_UPDATES, PRE_ECHO_THRESHOLD,
};
use crate::ring::DownsampledRenderBuffer;

/// 某个信号偏移下的 lag 估计（`MatchedFilter::LagEstimate`）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LagEstimate {
    /// 主峰延迟（降采样样本）。
    pub lag: usize,
    /// 前回声起点（降采样样本）。
    pub pre_echo_lag: usize,
}

/// 单个 lag 估计对应的延迟（供日志/测试）。
impl LagEstimate {
    pub fn new(lag: usize, pre_echo_lag: usize) -> Self {
        Self { lag, pre_echo_lag }
    }
}

/// 部分误差平滑：快降（立即跟随）/慢升（`UpdateAccumulatedError`，matched_filter.cc）。
fn update_accumulated_error(
    instantaneous: &[f32],
    accumulated: &mut [f32],
    one_over_error_sum_anchor: f32,
) {
    for k in 0..accumulated.len() {
        let error_norm = instantaneous[k] * one_over_error_sum_anchor;
        if error_norm < accumulated[k] {
            accumulated[k] = error_norm;
        } else {
            accumulated[k] += ACCUMULATED_ERROR_SMOOTH_UP * (error_norm - accumulated[k]);
        }
    }
}

/// 从峰值向更小延迟方向扫描前回声起点（`ComputePreEchoLag`，matched_filter.cc）。
fn compute_pre_echo_lag(
    accumulated_error: &[f32],
    lag: usize,
    alignment_shift_winner: usize,
) -> usize {
    debug_assert!(lag >= alignment_shift_winner);
    let mut pre_echo_lag_estimate = lag - alignment_shift_winner;
    let maximum_pre_echo_lag = (pre_echo_lag_estimate / ACCUMULATED_ERROR_SUB_SAMPLE_RATE)
        .min(accumulated_error.len());
    let mut k = maximum_pre_echo_lag as isize - 1;
    while k >= 0 {
        if accumulated_error[k as usize] > PRE_ECHO_THRESHOLD {
            break;
        }
        pre_echo_lag_estimate = ((k + 1) * ACCUMULATED_ERROR_SUB_SAMPLE_RATE as isize - 1) as usize;
        k -= 1;
    }
    pre_echo_lag_estimate + alignment_shift_winner
}

/// 求 |h[k]|² 最大的下标（`MaxSquarePeakIndex`）。
pub fn max_square_peak_index(h: &[f32]) -> usize {
    if h.len() < 2 {
        return 0;
    }
    let mut max_element1 = h[0] * h[0];
    let mut max_element2 = h[1] * h[1];
    let mut lag_estimate1 = 0usize;
    let mut lag_estimate2 = 1usize;
    let last_index = h.len() - 1;
    let mut k = 2;
    while k < last_index {
        let element1 = h[k] * h[k];
        let element2 = h[k + 1] * h[k + 1];
        if element1 > max_element1 {
            max_element1 = element1;
            lag_estimate1 = k;
        }
        if element2 > max_element2 {
            max_element2 = element2;
            lag_estimate2 = k + 1;
        }
        k += 2;
    }
    if max_element2 > max_element1 {
        max_element1 = max_element2;
        lag_estimate1 = lag_estimate2;
    }
    let last_element = h[last_index] * h[last_index];
    if last_element > max_element1 {
        last_index
    } else {
        lag_estimate1
    }
}

/// 标量核心（不含部分误差，对应源码 else 分支）。
fn matched_filter_core_plain(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
) {
    let x_size = x.len();
    for i in 0..y.len() {
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;
        let mut x_index = x_start_index;
        for k in 0..h.len() {
            x2_sum += x[x_index] * x[x_index];
            s += h[k] * x[x_index];
            x_index = if x_index < x_size - 1 { x_index + 1 } else { 0 };
        }

        let e = y[i] - s;
        let saturation = y[i] >= MATCHED_FILTER_SATURATION_LIMIT
            || y[i] <= -MATCHED_FILTER_SATURATION_LIMIT;
        *error_sum += e * e;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * e / x2_sum;
            let mut x_index2 = x_start_index;
            for k in 0..h.len() {
                h[k] += alpha * x[x_index2];
                x_index2 = if x_index2 < x_size - 1 { x_index2 + 1 } else { 0 };
            }
            *filters_updated = true;
        }

        x_start_index = if x_start_index > 0 { x_start_index - 1 } else { x_size - 1 };
    }
}

/// 标量核心（含部分误差，对应源码 `MatchedFilterCoreWithAccumulatedError` 分支）。
fn matched_filter_core_with_error(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
    accumulated_error: &mut [f32],
) {
    let x_size = x.len();
    accumulated_error.fill(0.0);
    for i in 0..y.len() {
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;
        let mut x_index = x_start_index;
        for k in 0..h.len() {
            x2_sum += x[x_index] * x[x_index];
            s += h[k] * x[x_index];
            x_index = if x_index < x_size - 1 { x_index + 1 } else { 0 };
            // 每 4 个抽头记录一次部分和误差（源码：`if ((k + 1 & 0b11) == 0)`）。
            if (k + 1) & 0b11 == 0 {
                let idx = k >> 2;
                accumulated_error[idx] += (y[i] - s) * (y[i] - s);
            }
        }

        let e = y[i] - s;
        let saturation = y[i] >= MATCHED_FILTER_SATURATION_LIMIT
            || y[i] <= -MATCHED_FILTER_SATURATION_LIMIT;
        *error_sum += e * e;

        if x2_sum > x2_sum_threshold && !saturation {
            let alpha = smoothing * e / x2_sum;
            let mut x_index2 = x_start_index;
            for k in 0..h.len() {
                h[k] += alpha * x[x_index2];
                x_index2 = if x_index2 < x_size - 1 { x_index2 + 1 } else { 0 };
            }
            *filters_updated = true;
        }

        x_start_index = if x_start_index > 0 { x_start_index - 1 } else { x_size - 1 };
    }
}

/// 递归更新的互相关估计器组（`MatchedFilter`）。
pub struct MatchedFilter {
    sub_block_size: usize,
    filter_intra_lag_shift: usize,
    filters: Vec<Vec<f32>>,
    /// 每 4 抽头一组的（平滑后）部分误差，仅 winner 滤波器维护。
    accumulated_error: Vec<Vec<f32>>,
    instantaneous_accumulated_error: Vec<f32>,
    reported_lag_estimate: Option<LagEstimate>,
    winner_lag: Option<usize>,
    last_detected_best_lag_filter: isize,
    number_pre_echo_updates: i32,
    excitation_limit: f32,
    smoothing_fast: f32,
    smoothing_slow: f32,
    matching_filter_threshold: f32,
    detect_pre_echo: bool,
}

impl MatchedFilter {
    /// 参数对应 `EchoPathDelayEstimator` 构造处的传递
    /// （echo_path_delay_estimator.cc:39-52）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sub_block_size: usize,
        window_size_sub_blocks: usize,
        num_matched_filters: usize,
        alignment_shift_sub_blocks: usize,
        excitation_limit: f32,
        smoothing_fast: f32,
        smoothing_slow: f32,
        matching_filter_threshold: f32,
        detect_pre_echo: bool,
    ) -> Self {
        let filter_size = window_size_sub_blocks * sub_block_size;
        let error_size = filter_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE;
        Self {
            sub_block_size,
            filter_intra_lag_shift: alignment_shift_sub_blocks * sub_block_size,
            filters: vec![vec![0.0; filter_size]; num_matched_filters],
            accumulated_error: vec![vec![1.0; error_size]; num_matched_filters],
            instantaneous_accumulated_error: vec![0.0; error_size],
            reported_lag_estimate: None,
            winner_lag: None,
            last_detected_best_lag_filter: -1,
            number_pre_echo_updates: 0,
            excitation_limit,
            smoothing_fast,
            smoothing_slow,
            matching_filter_threshold,
            detect_pre_echo,
        }
    }

    pub fn reset(&mut self, full_reset: bool) {
        for f in self.filters.iter_mut() {
            f.fill(0.0);
        }
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        if full_reset {
            for e in self.accumulated_error.iter_mut() {
                e.fill(1.0);
            }
            self.number_pre_echo_updates = 0;
        }
    }

    /// 最大可检测延迟（降采样样本，`GetMaxFilterLag`）= 5·384 + 512 = 2432。
    pub fn get_max_filter_lag(&self) -> usize {
        self.filters.len() * self.filter_intra_lag_shift + self.filters[0].len()
    }

    pub fn get_best_lag_estimate(&self) -> Option<LagEstimate> {
        self.reported_lag_estimate
    }

    pub fn filters(&self) -> &[Vec<f32>] {
        &self.filters
    }

    /// 用当前 capture 子块更新相关估计（`MatchedFilter::Update`）。
    pub fn update(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[f32],
        use_slow_smoothing: bool,
    ) {
        debug_assert_eq!(self.sub_block_size, capture.len());
        let y = capture;
        let x = &render_buffer.buffer;
        let x_size = x.len();

        let smoothing = if use_slow_smoothing {
            self.smoothing_slow
        } else {
            self.smoothing_fast
        };
        let x2_sum_threshold = self.filters[0].len() as f32
            * self.excitation_limit
            * self.excitation_limit;

        // 锚点：不做任何滤波时的基线误差能量。
        let mut error_sum_anchor = 0.0f32;
        for v in y.iter() {
            error_sum_anchor += v * v;
        }

        let mut winner_error_sum = error_sum_anchor;
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        let mut alignment_shift = 0usize;
        let mut previous_lag_estimate: Option<usize> = None;
        let num_filters = self.filters.len();
        let mut winner_index: isize = -1;
        for n in 0..num_filters {
            let mut error_sum = 0.0f32;
            let mut filters_updated = false;
            let compute_pre_echo =
                self.detect_pre_echo && n as isize == self.last_detected_best_lag_filter;

            let x_start_index =
                (render_buffer.read() as usize + alignment_shift + self.sub_block_size - 1)
                    % x_size;

            if compute_pre_echo {
                // 借出单条滤波器与瞬时误差缓冲（源码把瞬时误差作为成员传入）。
                let Self {
                    filters,
                    instantaneous_accumulated_error,
                    ..
                } = self;
                matched_filter_core_with_error(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    x,
                    y,
                    &mut filters[n],
                    &mut filters_updated,
                    &mut error_sum,
                    instantaneous_accumulated_error,
                );
            } else {
                let filters = &mut self.filters;
                matched_filter_core_plain(
                    x_start_index,
                    x2_sum_threshold,
                    smoothing,
                    x,
                    y,
                    &mut filters[n],
                    &mut filters_updated,
                    &mut error_sum,
                );
            }

            // 峰值位置 = 该滤波器覆盖范围内的延迟。
            let lag_estimate = max_square_peak_index(&self.filters[n]);
            let reliable = lag_estimate > MATCHED_FILTER_PEAK_LOW_MARGIN
                && lag_estimate < self.filters[n].len() - MATCHED_FILTER_PEAK_HIGH_MARGIN
                && error_sum < self.matching_filter_threshold * error_sum_anchor;

            let lag = lag_estimate + alignment_shift;
            if filters_updated && reliable && error_sum < winner_error_sum {
                winner_error_sum = error_sum;
                winner_index = n as isize;
                // 重叠区命中同一延迟时取下标更小（更早）的滤波器，
                // 为前回声检测保留扫描空间（源码注释）。
                if let Some(prev) = previous_lag_estimate {
                    if prev == lag {
                        self.winner_lag = Some(prev);
                        winner_index = n as isize - 1;
                    } else {
                        self.winner_lag = Some(lag);
                    }
                } else {
                    self.winner_lag = Some(lag);
                }
            }
            previous_lag_estimate = Some(lag);
            alignment_shift += self.filter_intra_lag_shift;
        }

        if winner_index != -1 {
            let winner_lag = self.winner_lag.unwrap();
            self.reported_lag_estimate = Some(LagEstimate::new(winner_lag, winner_lag));
            if self.detect_pre_echo && self.last_detected_best_lag_filter == winner_index {
                const ENERGY_THRESHOLD: f32 = 1.0;
                if error_sum_anchor > ENERGY_THRESHOLD {
                    let inst = self.instantaneous_accumulated_error.clone();
                    update_accumulated_error(
                        &inst,
                        &mut self.accumulated_error[winner_index as usize],
                        1.0 / error_sum_anchor,
                    );
                    self.number_pre_echo_updates += 1;
                }
                if self.number_pre_echo_updates >= PRE_ECHO_MIN_UPDATES {
                    let pre = compute_pre_echo_lag(
                        &self.accumulated_error[winner_index as usize],
                        winner_lag,
                        winner_index as usize * self.filter_intra_lag_shift,
                    );
                    self.reported_lag_estimate.as_mut().unwrap().pre_echo_lag = pre;
                } else {
                    self.reported_lag_estimate.as_mut().unwrap().pre_echo_lag = winner_lag;
                }
            }
            self.last_detected_best_lag_filter = winner_index;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{DELAY_SUB_BLOCK_SIZE, DOWNSAMPLED_BUFFER_SIZE};

    fn ds_buffer() -> DownsampledRenderBuffer {
        DownsampledRenderBuffer::new(DOWNSAMPLED_BUFFER_SIZE, DELAY_SUB_BLOCK_SIZE)
    }

    fn default_mf() -> MatchedFilter {
        use crate::constants as c;
        MatchedFilter::new(
            DELAY_SUB_BLOCK_SIZE,
            c::MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
            c::DELAY_NUM_FILTERS,
            c::MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
            c::POOR_EXCITATION_RENDER_LIMIT,
            0.7,
            0.7,
            c::DELAY_CANDIDATE_DETECTION_THRESHOLD,
            c::DELAY_DETECT_PRE_ECHO,
        )
    }

    /// T7: 强噪声 render + 延迟 L 的 capture → 多次更新后 winner 峰位 == L。
    #[test]
    fn converges_to_known_lag() {
        struct Xor(u64);
        impl Xor {
            fn next(&mut self) -> f32 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as f32 / (1u64 << 30) as f32
                    - 1.0)
                    * 2000.0
            }
        }
        for &lag in [10usize, 400, 1000].iter() {
            let mut mf = default_mf();
            let mut rb = ds_buffer();
            let mut rng = Xor(0x9E3779B97F4A7C15 ^ lag as u64);
            for v in rb.buffer.iter_mut() {
                *v = rng.next();
            }
            let size = DOWNSAMPLED_BUFFER_SIZE;
            // 模拟真实运行：read 每块前移 16（prepare_capture_processing 的效果），
            // y 取当前 read 窗口内延迟 lag 的数据 —— 每块都是新样本，
            // 回归设计矩阵超定，峰唯一收敛到 k = lag − n·384。
            // （若 y 固定不变，16 个方程解 512 个抽头会退化为最小范数解，伪峰胜出。）
            let mut winner: Option<usize> = None;
            for _ in 0..120usize {
                // 先推进 read（对应 prepare_capture_processing），再用新 read 取 y
                rb.update_read(-(DELAY_SUB_BLOCK_SIZE as isize));
                let read = rb.read();
                let mut cap = [0.0f32; DELAY_SUB_BLOCK_SIZE];
                for (i, v) in cap.iter_mut().enumerate() {
                    *v = rb.buffer[rb.offset_index(
                        read,
                        DELAY_SUB_BLOCK_SIZE as isize - 1 - i as isize + lag as isize,
                    ) as usize];
                }
                mf.update(&rb, &cap, false);
                if let Some(est) = mf.get_best_lag_estimate() {
                    winner = Some(est.lag);
                }
            }
            let got = winner.expect("120 次强激励更新后应有 lag 估计");
            assert_eq!(got, lag, "lag={} 检出={}", lag, got);
        }
    }

    /// 峰值检索：明确的最大值。
    #[test]
    fn peak_index() {
        let mut h = vec![0.0f32; 512];
        h[123] = 5.0;
        h[124] = -3.0;
        assert_eq!(max_square_peak_index(&h), 123);
        h[124] = 6.0;
        assert_eq!(max_square_peak_index(&h), 124);
        h[511] = 10.0;
        assert_eq!(max_square_peak_index(&h), 511);
    }

    /// 前回声扫描：从峰位向小延迟方向找第一个误差显著增大的组。
    #[test]
    fn pre_echo_scan() {
        // 情形 1：组 0..5 误差高（回声起点在组 5 之前），组 5 起被解释 →
        // 扫描在 k=4 处 break，起点保留 k=5 的值 (5+1)·4−1 = 23。
        let mut acc = vec![0.9f32; 128];
        for e in acc.iter_mut().skip(5) {
            *e = 0.1;
        }
        let pre = compute_pre_echo_lag(&acc, 500, 384);
        assert_eq!(pre, 23 + 384);

        // 情形 2：全程误差低于阈值 → 扫到 k=0，起点 = (0+1)·4−1 = 3。
        let acc2 = vec![0.1f32; 128];
        let pre2 = compute_pre_echo_lag(&acc2, 500, 384);
        assert_eq!(pre2, 3 + 384);

        // 情形 3：峰位下一组误差即高 → 立即 break，起点 = 峰位本身。
        let mut acc3 = vec![0.9f32; 128];
        acc3[10] = 0.1; // 不影响：扫描起点在 k=28
        let pre3 = compute_pre_echo_lag(&acc3, 500, 384);
        assert_eq!(pre3, 116 + 384);
    }
}
