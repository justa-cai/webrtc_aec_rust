//! 抑制滤波器（增益应用），对照
//! `modules/audio_processing/aec3/suppression_filter.{h,cc}`。
//!
//! 低带 WOLA：`E·G + √(1−G²)·N` → IFFT → 与上一块后半做 √Hann 加窗重叠相加。
//! 本实现 IFFT 已含 1/128 归一化（往返恒等），故无需源码的 2/128 因子。
//!
//! 高带（>8 kHz）：纯时域标量增益 + 延迟一块 + 仅 band1 加噪。16 kHz 单声道下
//! 只有 band 0，高带分支不激活（保留代码与源码对齐）。

use crate::constants::{
    BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, SUPPRESSION_CLAMP,
};
use crate::fft::Aec3Fft;
use crate::fft_data::FftData;

/// 抑制滤波器（`SuppressionFilter`），单声道。
pub struct SuppressionFilter {
    fft: Aec3Fft,
    /// 每带上一块输出（低带存 IFFT 后半，高带存上块样本）。
    e_output_old_band0: [f32; FFT_LENGTH_BY_2],
    // 复用缓冲
    noise_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    e_extended: [f32; FFT_LENGTH],
}

impl Default for SuppressionFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl SuppressionFilter {
    pub fn new() -> Self {
        Self {
            fft: Aec3Fft::new(),
            e_output_old_band0: [0.0; FFT_LENGTH_BY_2],
            noise_gain: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e_extended: [0.0; FFT_LENGTH],
        }
    }

    /// 应用抑制增益（`SuppressionFilter::ApplyGain`，16 kHz 单带版本）。
    ///
    /// `suppression_gain` 为**幅度域**增益（GetGain 输出已开方）；
    /// `e`（band 0）原地替换为输出。
    pub fn apply_gain(
        &mut self,
        comfort_noise: &FftData,
        suppression_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        e_spectrum: &FftData,
        e: &mut [f32; BLOCK_SIZE],
    ) {
        // 舒适噪声增益 √(1−g²)：信号与噪声功率互补。
        for i in 0..FFT_LENGTH_BY_2_PLUS_1 {
            self.noise_gain[i] =
                (1.0 - suppression_gain[i] * suppression_gain[i]).max(0.0).sqrt();
        }

        let mut spectrum = FftData::default();
        spectrum.assign(e_spectrum);
        for i in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let e_real = spectrum.re[i] * suppression_gain[i];
            let e_imag = spectrum.im[i] * suppression_gain[i];
            spectrum.re[i] = e_real + self.noise_gain[i] * comfort_noise.re[i];
            spectrum.im[i] = e_imag + self.noise_gain[i] * comfort_noise.im[i];
        }

        // 合成滤波器组（本实现 IFFT 已归一化）。
        self.fft.ifft(&spectrum, &mut self.e_extended);

        // WOLA：上一块后半（窗后半）+ 本块前半（窗前半）。
        for i in 0..FFT_LENGTH_BY_2 {
            let w_lo = crate::fft::sqrt_hanning(i);
            let w_hi = crate::fft::sqrt_hanning(FFT_LENGTH_BY_2 + i);
            let e0_i = self.e_output_old_band0[i] * w_hi + self.e_extended[i] * w_lo;
            e[i] = e0_i;
        }
        // 本块后半留待下一块。
        self.e_output_old_band0
            .copy_from_slice(&self.e_extended[FFT_LENGTH_BY_2..]);

        // 输出钳位（源码对所有带 SafeClamp −32768..32767）。
        for v in e.iter_mut() {
            *v = v.clamp(-SUPPRESSION_CLAMP, SUPPRESSION_CLAMP - 1.0);
        }
    }

    /// 高带处理接口（`ApplyGain` 的高带部分）。16 kHz 单带下 `bands` 为空即无操作；
    /// 保留以对齐源码语义：标量增益 → band1 加噪 → 延迟一块 → 钳位。
    pub fn apply_gain_high_bands(
        &mut self,
        _comfort_noise_high_band: &FftData,
        _high_bands_gain: f32,
        _bands: &mut [&mut [f32; BLOCK_SIZE]],
    ) {
        // 16 kHz 单声道实现无高带；多带支持留给后续（见 README 偏差清单）。
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{sqrt_hanning, Window};
    use crate::fft::Aec3Fft as Fft;

    /// 全 1 增益 + 零噪声：WOLA 输出应重建（√Hann 分析+合成，COLA 恒定）
    /// 输入信号的每块后半——即与"只做加窗变换再变换回来"的参考一致。
    #[test]
    fn unity_gain_reconstruction() {
        let mut sf = SuppressionFilter::new();
        let mut fft = Fft::new();

        // 用正弦输入逐块跑
        let mut x_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut max_err = 0.0f32;
        for t in 0..50 {
            let mut x = [0.0f32; BLOCK_SIZE];
            for (i, v) in x.iter_mut().enumerate() {
                *v = (2.0 * std::f32::consts::PI * 500.0
                    * (t * BLOCK_SIZE + i) as f32
                    / 16000.0)
                    .sin()
                    * 1000.0;
            }
            let mut spectrum = FftData::default();
            fft.padded_fft(&x, &x_old, Window::SqrtHanning, &mut spectrum);
            x_old.copy_from_slice(&x);

            let g = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let noise = FftData::default();
            let mut e = [0.0f32; BLOCK_SIZE];
            sf.apply_gain(&noise, &g, &spectrum, &mut e);

            if t >= 2 {
                // 稳态后（重叠历史就位）应精确重建输入
                for i in 0..BLOCK_SIZE {
                    max_err = max_err.max((e[i] - x[i]).abs());
                }
            }
        }
        // WebRTC 的 √Hann128 是 symmetric Hann，COLA 为近似（和 = 1+O(1.5e-4)）
        assert!(max_err < 1.0, "全增益重建误差 {} 过大", max_err);
    }

    /// 全 0 增益：输出应为纯舒适噪声（谱乘 √(1−0)=1）。
    #[test]
    fn zero_gain_passes_noise() {
        let mut sf = SuppressionFilter::new();
        let mut fft = Fft::new();
        let x = [1000.0f32; BLOCK_SIZE];
        let mut spectrum = FftData::default();
        fft.padded_fft(&x, &x, Window::SqrtHanning, &mut spectrum);

        // 构造平坦噪声谱
        let mut noise = FftData::default();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            noise.re[k] = 100.0;
            noise.im[k] = 50.0;
        }
        let g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut e = [0.0f32; BLOCK_SIZE];
        sf.apply_gain(&noise, &g, &spectrum, &mut e);
        // 第一个块输出 = 本块前半（噪声 IFFT 的前半·窗）+ 0（旧半为 0）
        // 信号被完全移除：与全 1 增益下的输出（信号）无关
        let mut e2 = [0.0f32; BLOCK_SIZE];
        sf.apply_gain(&noise, &g, &spectrum, &mut e2);
        // 非零（噪声注入）且有限
        assert!(e2.iter().all(|v| v.is_finite()));
        assert!(e2.iter().any(|v| v.abs() > 0.01));
    }

    /// 钳位生效。
    #[test]
    fn clamps_output() {
        let mut sf = SuppressionFilter::new();
        let mut spectrum = FftData::default();
        spectrum.re[0] = 1e7; // 巨大 DC
        let g = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let noise = FftData::default();
        let mut e = [0.0f32; BLOCK_SIZE];
        sf.apply_gain(&noise, &g, &spectrum, &mut e);
        assert!(e.iter().all(|v| *v >= -SUPPRESSION_CLAMP && *v <= SUPPRESSION_CLAMP - 1.0));
    }

    /// 窗函数与 fft.rs 的表一致（sqrt_hanning 访问器）。
    #[test]
    fn window_accessor() {
        assert!((sqrt_hanning(0)).abs() < 1e-6);
        assert!((sqrt_hanning(64) - 1.0).abs() < 1e-3); // symmetric 峰值 0.99992
        // symmetric 窗端点非零：[127] 与 [1] 对称
        assert!((sqrt_hanning(127) - 0.02454122852291).abs() < 1e-6);
        assert!((sqrt_hanning(127) - sqrt_hanning(1)).abs() < 1e-9);
        assert!((sqrt_hanning(1) - 0.02454122852291).abs() < 1e-7);
    }
}
