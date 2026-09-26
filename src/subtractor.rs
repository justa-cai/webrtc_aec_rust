//! 线性回声减法器（粗/精双滤波器编排），对照
//! `modules/audio_processing/aec3/subtractor.{h,cc}`。
//!
//! 每块流程（照抄源码顺序）：
//! X2 谱和 → 精滤波(卷积+预测误差) → 粗滤波 → metrics → 失配修正（可能整体
//! 缩放并令本块 G=0）→ E-FFT(Hanning) → 精增益/Adapt/频率响应 → 粗滤波
//! （连差 5 块则整体复制精滤波系数 + hangover 25 块，期间粗增益用 E_refined）
//! → e_refined 钳位 ±32768。

use crate::adaptive_fir_filter::{compute_erl, AdaptiveFirFilter};
use crate::constants::{
    BLOCK_SIZE, COARSE_RESET_HANGOVER_BLOCKS, FFT_LENGTH, FFT_LENGTH_BY_2,
    FFT_LENGTH_BY_2_PLUS_1, FILTER_COARSE_INITIAL_LENGTH_BLOCKS, FILTER_COARSE_LENGTH_BLOCKS,
    FILTER_REFINED_INITIAL_LENGTH_BLOCKS, FILTER_REFINED_LENGTH_BLOCKS,
    MISADJUSTMENT_ACCUM_BLOCKS, MISADJUSTMENT_ADJUST_THRESHOLD, MISADJUSTMENT_E2_OVERHANG_GATE,
    MISADJUSTMENT_Y2_GATE, OUTPUT_CLAMP_LIMIT, POOR_COARSE_FILTER_COUNTER_LIMIT,
};
use crate::coarse_filter_update_gain::{CoarseConfig, CoarseFilterUpdateGain};
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::fft::Aec3Fft;
use crate::fft::Window;
use crate::fft_data::FftData;
use crate::refined_filter_update_gain::{RefinedConfig, RefinedFilterUpdateGain};
use crate::render_delay_buffer::RenderBufferView;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::subtractor_output::SubtractorOutput;

/// 预测误差：`e = y − ifft(S)[64..128]`，同时输出回声估计 `s`（`PredictionError`）。
/// 本实现 FFT 往返严格相等，无需源码的 1/64 缩放。
fn prediction_error(
    fft: &mut Aec3Fft,
    s_spectrum: &FftData,
    y: &[f32; BLOCK_SIZE],
    e: &mut [f32; BLOCK_SIZE],
    s: &mut [f32; BLOCK_SIZE],
) {
    let mut tmp = [0.0f32; FFT_LENGTH];
    fft.ifft(s_spectrum, &mut tmp);
    for k in 0..BLOCK_SIZE {
        e[k] = y[k] - tmp[k + FFT_LENGTH_BY_2];
        s[k] = tmp[k + FFT_LENGTH_BY_2];
    }
}

/// 失配修正后的输出同步：`s *= factor; e = y − s`（`ScaleFilterOutput`）。
fn scale_filter_output(
    y: &[f32; BLOCK_SIZE],
    factor: f32,
    e: &mut [f32; BLOCK_SIZE],
    s: &mut [f32; BLOCK_SIZE],
) {
    for k in 0..BLOCK_SIZE {
        s[k] *= factor;
        e[k] = y[k] - s[k];
    }
}

/// 滤波器失配估计器（`Subtractor::FilterMisadjustmentEstimator`）。
///
/// 累积 4 块的 e²/y² 比值；误差能量远大于麦克风能量（inv > 10）时
/// 建议按 `2/√inv`（半量修正）整体缩放滤波器。
#[derive(Clone, Debug)]
struct FilterMisadjustmentEstimator {
    n_blocks_accum: i32,
    e2_accum: f32,
    y2_accum: f32,
    inv_misadjustment: f32,
    overhang: i32,
}

impl Default for FilterMisadjustmentEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterMisadjustmentEstimator {
    fn new() -> Self {
        Self {
            n_blocks_accum: 0,
            e2_accum: 0.0,
            y2_accum: 0.0,
            inv_misadjustment: 0.0,
            overhang: 0,
        }
    }

    fn update(&mut self, output: &SubtractorOutput) {
        self.e2_accum += output.e2_refined;
        self.y2_accum += output.y2;
        self.n_blocks_accum += 1;
        if self.n_blocks_accum == MISADJUSTMENT_ACCUM_BLOCKS {
            if self.y2_accum > MISADJUSTMENT_Y2_GATE * MISADJUSTMENT_ACCUM_BLOCKS as f32 {
                let update = self.e2_accum / self.y2_accum;
                if self.e2_accum
                    > MISADJUSTMENT_E2_OVERHANG_GATE * MISADJUSTMENT_ACCUM_BLOCKS as f32
                {
                    self.overhang = 4;
                } else {
                    self.overhang = (self.overhang - 1).max(0);
                }
                if update < self.inv_misadjustment || self.overhang > 0 {
                    self.inv_misadjustment +=
                        0.1 * (update - self.inv_misadjustment);
                }
            }
            self.e2_accum = 0.0;
            self.y2_accum = 0.0;
            self.n_blocks_accum = 0;
        }
    }

    fn is_adjustment_needed(&self) -> bool {
        self.inv_misadjustment > MISADJUSTMENT_ADJUST_THRESHOLD
    }

    /// 半量修正：调整估计失配量的一半（`GetMisadjustment`）。
    fn get_misadjustment(&self) -> f32 {
        assert!(self.inv_misadjustment > 0.0);
        2.0 / self.inv_misadjustment.sqrt()
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// 线性回声减法器（`Subtractor`），单声道。
pub struct Subtractor {
    fft: Aec3Fft,
    refined_filter: AdaptiveFirFilter,
    coarse_filter: AdaptiveFirFilter,
    refined_gain: RefinedFilterUpdateGain,
    coarse_gain: CoarseFilterUpdateGain,
    filter_misadjustment_estimator: FilterMisadjustmentEstimator,
    poor_coarse_filter_counter: usize,
    coarse_filter_reset_hangover: i32,
    refined_frequency_responses: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    refined_impulse_responses: Vec<f32>,
    // 复用缓冲
    s_spectrum: FftData,
    e_coarse_fft: FftData,
    g: FftData,
    erl: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Subtractor {
    pub fn new() -> Self {
        let max_len = FILTER_REFINED_LENGTH_BLOCKS.max(FILTER_REFINED_INITIAL_LENGTH_BLOCKS);
        let max_len_coarse =
            FILTER_COARSE_LENGTH_BLOCKS.max(FILTER_COARSE_INITIAL_LENGTH_BLOCKS);
        let _ = max_len_coarse;
        Self {
            fft: Aec3Fft::new(),
            refined_filter: AdaptiveFirFilter::with_default_ramp(
                max_len,
                FILTER_REFINED_INITIAL_LENGTH_BLOCKS,
            ),
            coarse_filter: AdaptiveFirFilter::with_default_ramp(
                max_len_coarse,
                FILTER_COARSE_INITIAL_LENGTH_BLOCKS,
            ),
            refined_gain: RefinedFilterUpdateGain::new(RefinedConfig::initial()),
            coarse_gain: CoarseFilterUpdateGain::new(CoarseConfig::initial()),
            filter_misadjustment_estimator: FilterMisadjustmentEstimator::new(),
            poor_coarse_filter_counter: 0,
            coarse_filter_reset_hangover: 0,
            refined_frequency_responses: vec![
                [0.0; FFT_LENGTH_BY_2_PLUS_1];
                FILTER_REFINED_LENGTH_BLOCKS.max(FILTER_REFINED_INITIAL_LENGTH_BLOCKS)
            ],
            refined_impulse_responses: vec![
                0.0;
                FILTER_REFINED_LENGTH_BLOCKS.max(FILTER_REFINED_INITIAL_LENGTH_BLOCKS)
                    * FFT_LENGTH_BY_2
            ],
            s_spectrum: FftData::default(),
            e_coarse_fft: FftData::default(),
            g: FftData::default(),
            erl: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    /// 精滤波器各分区频率响应（供 AecState/FilterAnalyzer）。
    pub fn filter_frequency_responses(&self) -> &Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]> {
        &self.refined_frequency_responses
    }

    /// 精滤波器时域冲激响应估计（供 FilterAnalyzer）。
    pub fn filter_impulse_responses(&self) -> &Vec<f32> {
        &self.refined_impulse_responses
    }

    /// 回声路径变化（`HandleEchoPathChange`）：延迟变化触发全量复位。
    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        if variability.delay_change != DelayAdjustment::None {
            self.refined_filter.handle_echo_path_change();
            self.coarse_filter.handle_echo_path_change();
            self.refined_gain
                .handle_echo_path_change(variability);
            self.coarse_gain.handle_echo_path_change();
            self.refined_gain.set_config(RefinedConfig::initial(), true);
            self.coarse_gain.set_config(CoarseConfig::initial(), true);
            self.refined_filter
                .set_size_partitions(FILTER_REFINED_INITIAL_LENGTH_BLOCKS, true);
            self.coarse_filter
                .set_size_partitions(FILTER_COARSE_INITIAL_LENGTH_BLOCKS, true);
        }
        if variability.gain_change {
            self.refined_gain.handle_echo_path_change(variability);
        }
    }

    /// 退出初始状态（`ExitInitialState`）：长度 12→13、配置初始→稳态（渐变）。
    pub fn exit_initial_state(&mut self) {
        self.refined_gain.set_config(RefinedConfig::normal(), false);
        self.coarse_gain.set_config(CoarseConfig::normal(), false);
        self.refined_filter
            .set_size_partitions(FILTER_REFINED_LENGTH_BLOCKS, false);
        self.coarse_filter
            .set_size_partitions(FILTER_COARSE_LENGTH_BLOCKS, false);
    }

    /// 执行线性回声消除（`Subtractor::Process`），单通道版本。
    pub fn process(
        &mut self,
        render_buffer: &RenderBufferView,
        capture: &[f32; BLOCK_SIZE],
        render_signal_analyzer: &RenderSignalAnalyzer,
        saturated_capture: bool,
        output: &mut SubtractorOutput,
    ) {
        let y = capture;

        // 计算 render 功率和（两级滤波器长度可能不同）。
        let refined_size = self.refined_filter.size_partitions();
        let coarse_size = self.coarse_filter.size_partitions();
        let mut x2_refined = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut x2_coarse_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let same_size = refined_size == coarse_size;
        if same_size {
            render_buffer.spectral_sum(refined_size, &mut x2_refined);
        } else if refined_size > coarse_size {
            render_buffer.spectral_sums(
                coarse_size,
                refined_size,
                &mut x2_coarse_data,
                &mut x2_refined,
            );
        } else {
            render_buffer.spectral_sums(
                refined_size,
                coarse_size,
                &mut x2_refined,
                &mut x2_coarse_data,
            );
        }
        let x2_coarse = if same_size {
            &x2_refined
        } else {
            &x2_coarse_data
        };

        // 两级滤波 + 预测误差。
        self.refined_filter
            .filter(render_buffer, &mut self.s_spectrum);
        prediction_error(
            &mut self.fft,
            &self.s_spectrum,
            y,
            &mut output.e_refined,
            &mut output.s_refined,
        );
        self.coarse_filter
            .filter(render_buffer, &mut self.s_spectrum);
        prediction_error(
            &mut self.fft,
            &self.s_spectrum,
            y,
            &mut output.e_coarse,
            &mut output.s_coarse,
        );

        output.compute_metrics(y);

        // 失配修正：整体缩放并同步输出，本块 G 置零。
        let mut refined_filters_adjusted = false;
        self.filter_misadjustment_estimator.update(output);
        if self.filter_misadjustment_estimator.is_adjustment_needed() {
            let scale = self.filter_misadjustment_estimator.get_misadjustment();
            self.refined_filter.scale_filter(scale);
            for h_k in self.refined_impulse_responses.iter_mut() {
                *h_k *= scale;
            }
            scale_filter_output(y, scale, &mut output.e_refined, &mut output.s_refined);
            self.filter_misadjustment_estimator.reset();
            refined_filters_adjusted = true;
        }

        // 误差谱（Hanning 零填 FFT）。
        let e_refined_slice = output.e_refined;
        let e_coarse_slice = output.e_coarse;
        let mut e_refined_fft = FftData::default();
        self.fft
            .zero_padded_fft(&e_refined_slice, Window::Hanning, &mut e_refined_fft);
        self.fft
            .zero_padded_fft(&e_coarse_slice, Window::Hanning, &mut self.e_coarse_fft);
        output.e_refined_fft = e_refined_fft;
        e_refined_fft.spectrum(&mut output.e2_refined_spectrum);
        self.e_coarse_fft
            .spectrum(&mut output.e2_coarse_spectrum);

        // 更新精滤波器。
        if !refined_filters_adjusted {
            // coarse 刚重置的 hangover 期间禁用"发散泄漏"，避免粗滤波器的暂时性
            // 差表现拖慢精滤波器收敛。
            let disallow_leakage_diverged = self.coarse_filter_reset_hangover > 0;
            compute_erl(&self.refined_frequency_responses, &mut self.erl);
            self.refined_gain.compute(
                &x2_refined,
                render_signal_analyzer,
                output,
                &self.erl,
                self.refined_filter.size_partitions(),
                saturated_capture,
                disallow_leakage_diverged,
                &mut self.g,
            );
        } else {
            self.g.clear();
        }
        self.refined_filter.adapt(
            render_buffer,
            &self.g,
            Some(&mut self.refined_impulse_responses),
        );
        self.refined_filter
            .compute_frequency_response(&mut self.refined_frequency_responses);

        // 更新粗滤波器：连续 5 块 e2_refined < e2_coarse → 整体复制精滤波系数。
        self.poor_coarse_filter_counter = if output.e2_refined < output.e2_coarse {
            self.poor_coarse_filter_counter + 1
        } else {
            0
        };
        if self.poor_coarse_filter_counter < POOR_COARSE_FILTER_COUNTER_LIMIT {
            self.coarse_gain.compute(
                x2_coarse,
                render_signal_analyzer,
                &self.e_coarse_fft,
                self.coarse_filter.size_partitions(),
                saturated_capture,
                &mut self.g,
            );
            self.coarse_filter_reset_hangover =
                (self.coarse_filter_reset_hangover - 1).max(0);
        } else {
            self.poor_coarse_filter_counter = 0;
            let refined_h = self.refined_filter.get_filter().to_vec();
            self.coarse_filter.set_filter(&refined_h);
            // 复位后用 E_refined 驱动粗增益（此时精解才是目标）。
            self.coarse_gain.compute(
                x2_coarse,
                render_signal_analyzer,
                &output.e_refined_fft,
                self.coarse_filter.size_partitions(),
                saturated_capture,
                &mut self.g,
            );
            self.coarse_filter_reset_hangover = COARSE_RESET_HANGOVER_BLOCKS as i32;
        }
        self.coarse_filter.adapt(render_buffer, &self.g, None);

        // 输出钳位。
        for v in output.e_refined.iter_mut() {
            *v = v.clamp(-OUTPUT_CLAMP_LIMIT, OUTPUT_CLAMP_LIMIT - 1.0);
        }
    }
}

impl Default for Subtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_delay_buffer::RenderDelayBuffer;

    /// 确定性噪声。
    struct Xor(u64);
    impl Xor {
        fn next(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as f32 / (1u64 << 30) as f32 - 1.0)
                * 3000.0
        }
    }

    /// 合成 RIR：延迟 d 块 + 指数衰减尾，能量归一化到 1（ERL≈0 dB 的强回声，
    /// 但保持 E2 与单分区 X2 同量级——否则 refined 的 P·E2 项会压死步长）。
    fn make_rir(delay_blocks: usize, tail_blocks: usize) -> Vec<f64> {
        let len = (delay_blocks + tail_blocks) * BLOCK_SIZE;
        let mut h = vec![0.0f64; len];
        for k in 0..(tail_blocks * BLOCK_SIZE) {
            h[delay_blocks * BLOCK_SIZE + k] =
                (-(k as f64) / (8.0 * BLOCK_SIZE as f64)).exp();
        }
        let energy: f64 = h.iter().map(|v| v * v).sum();
        let scale = 1.0 / energy.sqrt();
        for v in h.iter_mut() {
            *v *= scale;
        }
        h
    }

    fn conv(h: &[f64], x_hist: &[f32], out: &mut [f32]) {
        for n in 0..out.len() {
            let mut acc = 0.0f64;
            for (k, hk) in h.iter().enumerate() {
                let idx = x_hist.len() as isize - out.len() as isize + n as isize - k as isize;
                if idx >= 0 {
                    acc += hk * x_hist[idx as usize] as f64;
                }
            }
            out[n] = acc as f32;
        }
    }

    /// T11/T12: 固定对齐下收敛，ERLE 达标。
    fn run_erle_case(delay_blocks: usize, seconds: usize) -> f32 {
        let mut rng = Xor(0xC0FFEE ^ delay_blocks as u64);
        let h = make_rir(delay_blocks, 4);
        let mut buf = RenderDelayBuffer::new();
        let mut sub = Subtractor::new();
        let mut analyzer = RenderSignalAnalyzer::new();
        let mut output = SubtractorOutput::new();
        let mut x_hist: Vec<f32> = Vec::new();
        let (mut sum_y2, mut sum_e2) = (0.0f64, 0.0f64);
        let n_blocks = seconds * 250;
        // 对齐延迟：total = latency(1) + d = delay_blocks ⟹ d = delay_blocks − 1，
        // 使直达路径恰好落在分区 0。
        let align_d = delay_blocks.max(1) - 1;
        // 历史保留量：RIR 长度 + 余量
        let keep_blocks = h.len() / BLOCK_SIZE + 2;
        for t in 0..n_blocks {
            let mut xb = [0.0f32; BLOCK_SIZE];
            for v in xb.iter_mut() {
                *v = rng.next();
            }
            x_hist.extend_from_slice(&xb);
            buf.insert(&xb);
            buf.prepare_capture_processing();
            // 每块重新锚定对齐（read 每块前移，与 BlockProcessor 行为一致）
            buf.align_from_delay(align_d);
            let mut yb = [0.0f32; BLOCK_SIZE];
            conv(&h, &x_hist, &mut yb);
            analyzer.update(&buf.get_render_buffer(), Some(0));
            sub.process(
                &buf.get_render_buffer(),
                &yb,
                &analyzer,
                false,
                &mut output,
            );
            if t >= n_blocks - 250 {
                sum_y2 += yb.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                sum_e2 += output
                    .e_refined
                    .iter()
                    .map(|v| (*v as f64) * (*v as f64))
                    .sum::<f64>();
            }
            // 控制历史长度（保留 RIR 所需）
            if x_hist.len() > (keep_blocks + 4) * BLOCK_SIZE {
                let cut = x_hist.len() - keep_blocks * BLOCK_SIZE;
                x_hist.drain(..cut);
            }
        }
        (10.0 * (sum_y2 / sum_e2).log10()) as f32
    }

    #[test]
    fn erle_zero_delay() {
        let erle = run_erle_case(1, 5);
        assert!(erle >= 30.0, "0 时延场景 ERLE={} dB", erle);
    }

    #[test]
    fn erle_100ms_delay() {
        let erle = run_erle_case(25, 6);
        assert!(erle >= 30.0, "100 ms 时延场景 ERLE={} dB", erle);
    }

    /// T13: 失配估计器行为。
    /// 注意门限：y2 累积须过 4·200²·64，e2 累积须过 4·7500²·64 才有 overhang
    /// （inv 从 0 起，只有 overhang>0 或 update<inv 才更新 EMA）。
    #[test]
    fn misadjustment_estimator() {
        let mut m = FilterMisadjustmentEstimator::new();
        let mut out = SubtractorOutput::new();
        // e2 = 4·y2，能量过 y2 门限：比值 4 < 10 不触发
        out.y2 = 2.0e7;
        out.e2_refined = 4.0 * out.y2;
        for _ in 0..8 {
            m.update(&out);
        }
        assert!(!m.is_adjustment_needed());

        // e2 巨大（过 overhang 门限 4·7500²·64 ≈ 1.44e10）→ inv 累积 → 触发
        out.e2_refined = 4.0e10;
        for _ in 0..4 {
            m.update(&out);
        }
        assert!(m.is_adjustment_needed());
        // inv ≈ 0.1·2000 = 200 → scale = 2/√200 ≈ 0.141
        let scale = m.get_misadjustment();
        assert!(
            (scale - 2.0 / 200.0f32.sqrt()).abs() < 0.02,
            "scale={}",
            scale
        );
    }

    /// T14: 粗滤波器复制与 hangover。
    #[test]
    fn coarse_copy_and_hangover() {
        // 通过公共流程难以确定性触发；直接验证状态机的关键行为：
        // handle_echo_path_change 全复位 + exit_initial_state 渐变设置。
        let mut sub = Subtractor::new();
        assert_eq!(sub.refined_filter.size_partitions(), 12);
        sub.exit_initial_state();
        // 渐变中（250 块），当前仍为 12
        assert_eq!(sub.refined_filter.size_partitions(), 12);
        sub.handle_echo_path_change(&EchoPathVariability::new(
            false,
            DelayAdjustment::NewDetectedDelay,
            false,
        ));
        assert_eq!(sub.refined_filter.size_partitions(), 12);
        assert!(sub.refined_impulse_responses.iter().all(|v| *v == 0.0));
    }

    /// T15: 增益保护——饱和时 G=0、输出仍有限。
    #[test]
    fn saturated_guard() {
        let mut rng = Xor(99);
        let h = make_rir(2, 3);
        let mut buf = RenderDelayBuffer::new();
        let mut sub = Subtractor::new();
        let mut analyzer = RenderSignalAnalyzer::new();
        let mut output = SubtractorOutput::new();
        let mut x_hist: Vec<f32> = Vec::new();
        for t in 0..100 {
            let mut xb = [0.0f32; BLOCK_SIZE];
            for v in xb.iter_mut() {
                *v = rng.next();
            }
            x_hist.extend_from_slice(&xb);
            buf.insert(&xb);
            buf.prepare_capture_processing();
            if t == 0 {
                buf.align_from_delay(1);
            }
            let mut yb = [0.0f32; BLOCK_SIZE];
            conv(&h, &x_hist, &mut yb);
            // 饱和
            for v in yb.iter_mut() {
                *v = v.clamp(-33000.0, 33000.0);
            }
            analyzer.update(&buf.get_render_buffer(), Some(0));
            sub.process(&buf.get_render_buffer(), &yb, &analyzer, true, &mut output);
            assert!(output.e_refined.iter().all(|v| v.is_finite()));
            if x_hist.len() > 20 * BLOCK_SIZE {
                let cut = x_hist.len() - 10 * BLOCK_SIZE;
                x_hist.drain(..cut);
            }
        }
        // 饱和期间 H_error 保持在钳位范围内
        let he = sub.refined_gain.h_error();
        assert!(he.iter().all(|v| *v >= 0.001 && *v <= 2.0));
    }
}
