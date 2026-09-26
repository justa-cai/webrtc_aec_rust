//! ML 残余回声估计接口（trait 注入点），对照
//! `api/audio/neural_residual_echo_estimator.h`。
//!
//! 本实现不含 tflite 推理；注入实现后经 `ResidualEchoEstimator` 的单一调用点
//! 接入（与 C++ 耦合面一致）。三状态机由 `ResidualEchoEstimator` 持有：
//! 未注入 = Uninitialized（永远走传统路径）；注入且 `is_initialized` 且
//! `UsableLinearEstimate` → Active（模型输出直接覆盖 R2/R2_unbounded）。

use crate::constants::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1, SuppressorConfig};

/// ML 残余回声估计器接口（`NeuralResidualEchoEstimator`）。
pub trait NeuralResidualEchoEstimator {
    /// 模型是否已初始化（可推理）。
    fn is_initialized(&self) -> bool;

    /// 估计残余回声：输出逐 bin 功率谱，直接写入 R2/R2_unbounded。
    fn estimate(
        &mut self,
        render_block: &[f32; BLOCK_SIZE],
        capture: &[f32; BLOCK_SIZE],
        linear_aec_output: &[f32; BLOCK_SIZE],
        s2_linear: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        e2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        dominant_nearend: bool,
        r2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        r2_unbounded: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    );

    /// ML 激活时改写抑制器配置（`AdjustConfig`）。
    fn adjust_config(&self, config: SuppressorConfig) -> SuppressorConfig;
}

/// ML 残余估计状态机（`MlReeState`，residual_echo_estimator.h:60-68）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlReeState {
    /// 未注入模型（传统路径）。
    Uninitialized,
    /// 已注入且初始化，但线性估计不可用。
    Initialized,
    /// 激活：模型输出接管 R2。
    Active,
}
