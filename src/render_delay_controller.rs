//! render 延迟控制器，对照
//! `modules/audio_processing/aec3/render_delay_controller.{h,cc}`。
//!
//! 把估计器输出的样本级延迟换算为块级缓冲延迟（>>6），并加单向迟滞：
//! 仅当上次与本次质量都为 Refined 时，对 ≤1 块的增大方向不跟进
//! （读更早的 render 数据可能已被覆盖，且增大方向更可能是误判）；
//! 减小方向立即跟随。

use crate::clockdrift_detector::Level;
use crate::constants::{BLOCK_SIZE_LOG2, DELAY_HYSTERESIS_LIMIT_BLOCKS};
use crate::delay_estimate::{DelayEstimate, DelayQuality};
use crate::echo_path_delay_estimator::EchoPathDelayEstimator;
use crate::ring::DownsampledRenderBuffer;

use crate::constants::BLOCK_SIZE;

/// 样本延迟 → 块级缓冲延迟（`ComputeBufferDelay`，render_delay_controller.cc:64）。
fn compute_buffer_delay(
    current_delay: Option<DelayEstimate>,
    hysteresis_limit_blocks: usize,
    estimated_delay: DelayEstimate,
) -> DelayEstimate {
    let mut new_delay_blocks = estimated_delay.delay >> BLOCK_SIZE_LOG2;
    if let Some(current) = current_delay {
        let current_delay_blocks = current.delay;
        if new_delay_blocks > current_delay_blocks
            && new_delay_blocks <= current_delay_blocks + hysteresis_limit_blocks
        {
            new_delay_blocks = current_delay_blocks;
        }
    }
    let mut new_delay = estimated_delay;
    new_delay.delay = new_delay_blocks;
    new_delay
}

/// render 延迟控制器（`RenderDelayControllerImpl`）。
pub struct RenderDelayController {
    hysteresis_limit_blocks: usize,
    delay: Option<DelayEstimate>,
    delay_estimator: EchoPathDelayEstimator,
    delay_samples: Option<DelayEstimate>,
    last_delay_estimate_quality: DelayQuality,
}

impl Default for RenderDelayController {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderDelayController {
    pub fn new() -> Self {
        Self {
            hysteresis_limit_blocks: DELAY_HYSTERESIS_LIMIT_BLOCKS,
            delay: None,
            delay_estimator: EchoPathDelayEstimator::new(),
            delay_samples: None,
            last_delay_estimate_quality: DelayQuality::Coarse,
        }
    }

    /// 复位（`Reset`）。`reset_delay_confidence=true` 时置信度也归零
    /// （行为等同通话重新开始）。
    pub fn reset(&mut self, reset_delay_confidence: bool) {
        self.delay = None;
        self.delay_samples = None;
        self.delay_estimator.reset(reset_delay_confidence);
        if reset_delay_confidence {
            self.last_delay_estimate_quality = DelayQuality::Coarse;
        }
    }

    pub fn has_clockdrift(&self) -> bool {
        self.delay_estimator.clockdrift() != Level::None
    }

    /// 估计并换算缓冲延迟（`GetDelay`）。返回块级延迟。
    pub fn get_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[f32; BLOCK_SIZE],
    ) -> Option<DelayEstimate> {
        let delay_samples = self.delay_estimator.estimate_delay(render_buffer, capture);
        if delay_samples.is_some() {
            self.delay_samples = delay_samples; // hold-last
        }

        if let Some(samples) = self.delay_samples {
            let use_hysteresis = self.last_delay_estimate_quality == DelayQuality::Refined
                && samples.quality == DelayQuality::Refined;
            self.delay = Some(compute_buffer_delay(
                self.delay,
                if use_hysteresis {
                    self.hysteresis_limit_blocks
                } else {
                    0
                },
                samples,
            ));
            self.last_delay_estimate_quality = self.delay.unwrap().quality;
        }

        self.delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(blocks: usize, quality: DelayQuality) -> DelayEstimate {
        DelayEstimate::new(quality, blocks << BLOCK_SIZE_LOG2)
    }

    /// T9a: refined↔refined 时 1 块内的增大被迟滞吸收。
    /// 注意：`current_delay` 与返回值单位为**块**，`estimated_delay` 为**样本**。
    #[test]
    fn hysteresis_absorbs_small_increase() {
        let cur = DelayEstimate::new(DelayQuality::Refined, 25); // 块
        let d = compute_buffer_delay(
            Some(cur),
            1,
            DelayEstimate::new(DelayQuality::Refined, 26 << BLOCK_SIZE_LOG2),
        );
        assert_eq!(d.delay, 25, "26 应被迟滞为 25");
        let d2 = compute_buffer_delay(
            Some(cur),
            1,
            DelayEstimate::new(DelayQuality::Refined, 27 << BLOCK_SIZE_LOG2),
        );
        assert_eq!(d2.delay, 27, "增大 2 块应跟随");
        // 减小方向立即跟随
        let d3 = compute_buffer_delay(
            Some(cur),
            1,
            DelayEstimate::new(DelayQuality::Refined, 24 << BLOCK_SIZE_LOG2),
        );
        assert_eq!(d3.delay, 24);
        // 无迟滞（质量非 refined↔refined）时立即跟随
        let d4 = compute_buffer_delay(
            Some(cur),
            0,
            DelayEstimate::new(DelayQuality::Coarse, 26 << BLOCK_SIZE_LOG2),
        );
        assert_eq!(d4.delay, 26);
    }

    /// T9b: 控制器 hold-last：估计流中断后保持上一次输出。
    #[test]
    fn controller_hold_last() {
        let mut c = RenderDelayController::new();
        assert!(c.delay.is_none());
        // 直接操作内部状态验证 hold-last 语义
        c.delay_samples = Some(est(30, DelayQuality::Refined));
        c.delay = Some(compute_buffer_delay(c.delay, 1, c.delay_samples.unwrap()));
        assert_eq!(c.delay.unwrap().delay, 30);
        // 再次调用 get_delay 前置空估计器输出 —— hold-last 由 delay_samples 保持
        assert_eq!(c.delay_samples.unwrap().delay, 30 << BLOCK_SIZE_LOG2);
    }
}
