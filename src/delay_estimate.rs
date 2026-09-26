//! 时延估计结果，对照 `modules/audio_processing/aec3/delay_estimate.h`。

/// 估计质量等级（`DelayEstimate::Quality`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayQuality {
    /// 冷启动阶段的大胆猜测（直方图计数 > initial 门限）。
    Coarse,
    /// 已充分确认（直方图计数 > converged 门限）。
    Refined,
}

/// 时延估计（`DelayEstimate`）。`delay` 的单位随所处阶段不同：
/// 聚合器输出为降采样样本，乘以 `down_sampling_factor` 后为全带样本，
/// 控制器再 >>6 换算为块。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelayEstimate {
    pub quality: DelayQuality,
    pub delay: usize,
}

impl DelayEstimate {
    pub fn new(quality: DelayQuality, delay: usize) -> Self {
        Self { quality, delay }
    }
}
