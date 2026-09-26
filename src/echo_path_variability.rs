//! 回声路径变化描述，对照 `modules/audio_processing/aec3/echo_path_variability.h`。

/// 延迟调整类型（`EchoPathVariability::DelayAdjustment`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayAdjustment {
    /// 无变化。
    None,
    /// render 缓冲溢出触发的整体复位。
    BufferFlush,
    /// 检测到新的时延（触发线性滤波器全量重置）。
    NewDetectedDelay,
}

/// 回声路径变化（`EchoPathVariability`）。
///
/// 注意：`clock_drift` 仅为信息位，**不构成** `audio_path_changed()`
/// （与源码一致——漂移只影响抑制器策略，本移植中为惰性标志）。
#[derive(Clone, Copy, Debug)]
pub struct EchoPathVariability {
    pub gain_change: bool,
    pub delay_change: DelayAdjustment,
    pub clock_drift: bool,
}

impl EchoPathVariability {
    pub fn new(gain_change: bool, delay_change: DelayAdjustment, clock_drift: bool) -> Self {
        Self {
            gain_change,
            delay_change,
            clock_drift,
        }
    }

    /// 路径是否发生变化（需要通知滤波器复位等）。
    pub fn audio_path_changed(&self) -> bool {
        self.gain_change || self.delay_change != DelayAdjustment::None
    }
}
