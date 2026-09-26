//! Subtractor 输出结构，对照 `modules/audio_processing/aec3/subtractor_output.{h,cc}`。

use crate::constants::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1};
use crate::fft_data::FftData;

/// 单通道的减法器输出（`SubtractorOutput`）。
#[derive(Clone, Debug)]
pub struct SubtractorOutput {
    /// 精滤波器的回声估计（时域）。
    pub s_refined: [f32; BLOCK_SIZE],
    /// 粗滤波器的回声估计（时域）。
    pub s_coarse: [f32; BLOCK_SIZE],
    /// 精滤波器输出（y − s_refined）。
    pub e_refined: [f32; BLOCK_SIZE],
    /// 粗滤波器输出（y − s_coarse）。
    pub e_coarse: [f32; BLOCK_SIZE],
    /// e_refined 的 Hanning 零填 FFT（E_refined）。
    pub e_refined_fft: FftData,
    /// |E_refined[k]|²。
    pub e2_refined_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// |E_coarse[k]|²。
    pub e2_coarse_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// 时域能量统计。
    pub e2_refined: f32,
    pub e2_coarse: f32,
    pub s2_refined: f32,
    pub s2_coarse: f32,
    pub y2: f32,
    /// max|s|（饱和检测用）。
    pub s_refined_max_abs: f32,
    pub s_coarse_max_abs: f32,
}

impl Default for SubtractorOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl SubtractorOutput {
    pub fn new() -> Self {
        Self {
            s_refined: [0.0; BLOCK_SIZE],
            s_coarse: [0.0; BLOCK_SIZE],
            e_refined: [0.0; BLOCK_SIZE],
            e_coarse: [0.0; BLOCK_SIZE],
            e_refined_fft: FftData::default(),
            e2_refined_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e2_coarse_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e2_refined: 0.0,
            e2_coarse: 0.0,
            s2_refined: 0.0,
            s2_coarse: 0.0,
            y2: 0.0,
            s_refined_max_abs: 0.0,
            s_coarse_max_abs: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// 计算时域能量与峰值统计（`SubtractorOutput::ComputeMetrics`）。
    pub fn compute_metrics(&mut self, y: &[f32; BLOCK_SIZE]) {
        self.y2 = y.iter().map(|v| v * v).sum();
        self.e2_refined = self.e_refined.iter().map(|v| v * v).sum();
        self.e2_coarse = self.e_coarse.iter().map(|v| v * v).sum();
        self.s2_refined = self.s_refined.iter().map(|v| v * v).sum();
        self.s2_coarse = self.s_coarse.iter().map(|v| v * v).sum();
        self.s_refined_max_abs = self
            .s_refined
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        self.s_coarse_max_abs = self
            .s_coarse
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
    }
}
