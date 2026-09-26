//! 回声路径时延估计器，对照
//! `modules/audio_processing/aec3/echo_path_delay_estimator.{h,cc}`。
//!
//! 串联：capture 降采样 → 匹配滤波器组 → lag 聚合 → ×down_sampling_factor。
//! 延迟估计连续 0.5 s 不变时做软复位（清匹配滤波器 h、保留部分误差/直方图/
//! 置信度），让后续跟踪更灵敏。
//!
//! 单声道：capture 对齐混音 = 直接取 0 通道（上游 AlignmentMixer 对
//! num_channels==1 走 kFixed 分支）。

use crate::clockdrift_detector::{ClockdriftDetector, Level};
use crate::constants::{
    BLOCK_SIZE, DELAY_CANDIDATE_DETECTION_THRESHOLD, DELAY_DETECT_PRE_ECHO,
    DELAY_DOWN_SAMPLING_FACTOR, DELAY_ESTIMATE_SMOOTHING_FAST, DELAY_ESTIMATE_SMOOTHING_SLOW,
    DELAY_NUM_FILTERS, DELAY_SUB_BLOCK_SIZE, MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS, NUM_BLOCKS_PER_SECOND,
    POOR_EXCITATION_RENDER_LIMIT,
};
use crate::delay_estimate::{DelayEstimate, DelayQuality};
use crate::decimator::Decimator;
use crate::matched_filter::MatchedFilter;
use crate::matched_filter_lag_aggregator::MatchedFilterLagAggregator;
use crate::ring::DownsampledRenderBuffer;

/// 回声路径时延估计器（`EchoPathDelayEstimator`）。
pub struct EchoPathDelayEstimator {
    down_sampling_factor: usize,
    sub_block_size: usize,
    capture_decimator: Decimator,
    matched_filter: MatchedFilter,
    matched_filter_lag_aggregator: MatchedFilterLagAggregator,
    old_aggregated_lag: Option<DelayEstimate>,
    consistent_estimate_counter: usize,
    clockdrift_detector: ClockdriftDetector,
}

impl EchoPathDelayEstimator {
    pub fn new() -> Self {
        let matched_filter = MatchedFilter::new(
            DELAY_SUB_BLOCK_SIZE,
            MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
            DELAY_NUM_FILTERS,
            MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
            POOR_EXCITATION_RENDER_LIMIT,
            DELAY_ESTIMATE_SMOOTHING_FAST,
            DELAY_ESTIMATE_SMOOTHING_SLOW,
            DELAY_CANDIDATE_DETECTION_THRESHOLD,
            DELAY_DETECT_PRE_ECHO,
        );
        Self {
            down_sampling_factor: DELAY_DOWN_SAMPLING_FACTOR,
            sub_block_size: DELAY_SUB_BLOCK_SIZE,
            capture_decimator: Decimator::new(DELAY_DOWN_SAMPLING_FACTOR),
            matched_filter_lag_aggregator: MatchedFilterLagAggregator::new(
                matched_filter.get_max_filter_lag(),
            ),
            matched_filter,
            old_aggregated_lag: None,
            consistent_estimate_counter: 0,
            clockdrift_detector: ClockdriftDetector::new(),
        }
    }

    pub fn reset(&mut self, reset_delay_confidence: bool) {
        self.reset_impl(true, reset_delay_confidence);
    }

    /// 双参复位（`EchoPathDelayEstimator::Reset(bool, bool)`）。
    ///
    /// - `reset_lag_aggregator=true`：lag 聚合器复位（hard 与否由置信度参数决定）
    ///   且匹配滤波器 full_reset（清 h 与部分误差）。
    /// - `reset_lag_aggregator=false`（软复位）：仅清匹配滤波器 h（`full_reset=false`
    ///   保留部分误差与前回声计数），**不动**直方图与 `significant_candidate_found`。
    fn reset_impl(&mut self, reset_lag_aggregator: bool, reset_delay_confidence: bool) {
        if reset_lag_aggregator {
            self.matched_filter_lag_aggregator
                .reset(reset_delay_confidence);
        }
        self.matched_filter.reset(reset_lag_aggregator);
        self.old_aggregated_lag = None;
        self.consistent_estimate_counter = 0;
    }

    pub fn clockdrift(&self) -> Level {
        self.clockdrift_detector.level()
    }

    /// 估计一次时延（`EstimateDelay`）。输入全带 capture 块，输出全带样本单位。
    pub fn estimate_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[f32; BLOCK_SIZE],
    ) -> Option<DelayEstimate> {
        debug_assert_eq!(BLOCK_SIZE / self.down_sampling_factor, self.sub_block_size);

        // 单声道：混音 = 直通；降采样到 4 kHz 域。
        let mut downsampled_capture_data = [0.0f32; BLOCK_SIZE];
        let downsampled_capture =
            &mut downsampled_capture_data[..self.sub_block_size];
        self.capture_decimator.decimate(capture, downsampled_capture);

        self.matched_filter.update(
            render_buffer,
            &downsampled_capture_data[..self.sub_block_size],
            self.matched_filter_lag_aggregator.reliable_delay_found(),
        );

        let mut aggregated = self
            .matched_filter_lag_aggregator
            .aggregate(self.matched_filter.get_best_lag_estimate());

        // 时钟漂移检测：只喂 refined 级估计，且用 ×4 前的 ds 单位值。
        if let Some(d) = aggregated {
            if d.quality == DelayQuality::Refined {
                self.clockdrift_detector
                    .update(self.matched_filter_lag_aggregator.get_delay_at_highest_peak()
                        as i32);
            }
        }

        // 降采样域 lag → 全带样本。
        if let Some(d) = aggregated.as_mut() {
            d.delay *= self.down_sampling_factor;
        }

        // 一致估计 0.5 s → 软复位（已锁定，清历史让跟踪更灵敏）。
        match (self.old_aggregated_lag, aggregated) {
            (Some(old), Some(new)) if old.delay == new.delay => {
                self.consistent_estimate_counter += 1;
            }
            _ => {
                self.consistent_estimate_counter = 0;
            }
        }
        self.old_aggregated_lag = aggregated;
        if self.consistent_estimate_counter > NUM_BLOCKS_PER_SECOND / 2 {
            self.reset_impl(false, false);
        }

        aggregated
    }
}

impl Default for EchoPathDelayEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{BLOCK_SIZE, NUM_BLOCKS_PER_SECOND};
    use crate::render_delay_buffer::RenderDelayBuffer;
    use crate::render_delay_controller::RenderDelayController;

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

    /// M1 端到端：render WGN + 纯延迟 D 的 capture → 全链在 1.5 s 内锁定。
    ///
    /// 注意 AEC3 语义：报告的 delay 不是绝对延迟，而是让回声直达路径落在
    /// 线性滤波器覆盖范围**内部**的缓冲延迟——总对齐量 = delay + buffer_latency，
    /// 且 headroom（32 样本）故意少补偿一点。因此断言
    /// `(delay + latency)·64 − D ∈ [−64, 192]`（≈ headroom ± 1 块量化）。
    /// D=0 不现实（直达峰会落在 tap 边界被可靠性判据拒绝），最小取 256 样本。
    #[test]
    fn end_to_end_delay_lock() {
        for &delay_samples in [256usize, 1600, 4000, 8000].iter() {
            let mut rng = Xor(0xDEADBEEF ^ delay_samples as u64);
            let n = 400 * BLOCK_SIZE; // 400 块 = 1.6 s
            // capture[s] = render[s − D]（s < D 时为 0，模拟回声尚未到达）
            let mut render_stream: Vec<f32> = Vec::new();
            let mut capture_stream: Vec<f32> = Vec::new();
            for s in 0..n {
                let v = rng.next();
                render_stream.push(v);
                capture_stream.push(if s >= delay_samples {
                    render_stream[s - delay_samples]
                } else {
                    0.0
                });
            }

            let mut buf = RenderDelayBuffer::new();
            let mut controller = RenderDelayController::new();
            let mut last_delay: Option<usize> = None;
            let mut lock_block: Option<usize> = None;
            let mut refined_stable_since: Option<usize> = None;
            let mut changes = 0usize;

            for t in 0..400usize {
                let mut rblock = [0.0f32; BLOCK_SIZE];
                rblock.copy_from_slice(&render_stream[t * BLOCK_SIZE..(t + 1) * BLOCK_SIZE]);
                buf.insert(&rblock);
                buf.prepare_capture_processing();

                let mut cblock = [0.0f32; BLOCK_SIZE];
                cblock.copy_from_slice(&capture_stream[t * BLOCK_SIZE..(t + 1) * BLOCK_SIZE]);
                if let Some(d) = controller.get_delay(buf.get_downsampled_render_buffer(), &cblock)
                {
                    if buf.align_from_delay(d.delay) {
                        changes += 1;
                    }
                    last_delay = Some(d.delay);
                    // 锁定判定：首次达到 Refined 即视为锁定
                    if lock_block.is_none() && d.quality == DelayQuality::Refined {
                        lock_block = Some(t);
                        refined_stable_since = Some(t);
                    }
                }
            }

            let final_delay = last_delay.expect("应有延迟输出");
            let total_samples =
                ((final_delay + buf.buffer_latency()) * BLOCK_SIZE) as isize;
            let diff = total_samples - delay_samples as isize;
            assert!(
                (-64..=192).contains(&diff),
                "D={} 最终={} 块, total={} 样本, 偏差 {}",
                delay_samples,
                final_delay,
                total_samples,
                diff
            );
            let lock = lock_block.expect("应在 1.6 s 内锁定");
            assert!(
                lock < (NUM_BLOCKS_PER_SECOND + NUM_BLOCKS_PER_SECOND / 2),
                "D={} 锁定过慢: 块 {}",
                delay_samples,
                lock
            );
            // Refined 应出现且最终稳定
            assert!(
                refined_stable_since.is_some(),
                "D={} 应达到 Refined 质量",
                delay_samples
            );
            // 对齐变化次数有限（迟滞生效）
            assert!(changes <= 4, "D={} 对齐变化 {} 次过多", delay_samples, changes);
        }
    }
}
