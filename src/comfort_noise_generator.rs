//! 舒适噪声发生器，对照
//! `modules/audio_processing/aec3/comfort_noise_generator.{h,cc}`。
//!
//! 两部分：
//! 1. **N2 噪声底估计**（最小统计 + 缓慢上漂）：`Y2_smoothed` 一阶低通（0.1），
//!    前 50 块只攒平滑；N2 min 跟踪（0.9/0.1）+ 每块 ×1.0002 上漂；
//!    启动 1000 块内另有更慢（α=0.001）的 `N2_initial`；饱和时冻结；
//!    两者的每 bin 下限 `noise_floor ≈ 17.13`。
//! 2. **噪声生成**：LCG 随机相位 + `√2·sin(2πi/32)` 查找表；低带按 √N2 谱成形，
//!    高带用上半频段均值平谱。√2 补偿 WOLA 交叉衰减在不相关随机相位下的功率损失。

use crate::constants::{
    CNG_LCG_MULTIPLIER, CNG_N2_COUNTER_THRESHOLD, CNG_N2_INITIAL, CNG_N2_INITIAL_ALPHA,
    CNG_N2_INITIAL_BLOCKS, CNG_N2_MIN_UPDATE, CNG_N2_UPWARD_DRIFT, CNG_SEED,
    CNG_Y2_SMOOTHING, COMFORT_NOISE_FLOOR_DBFS, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1,
};
use crate::fft_data::FftData;

/// `√2·sin(2πi/32)` 查找表（comfort_noise_generator.cc:45-57，照抄）。
const SQRT2_SIN: [f32; 32] = [
    0.0000000,
    0.2758994,
    0.5411961,
    0.7856950,
    1.0000000,
    1.1758756,
    1.3065630,
    1.3870398,
    1.4142135,
    1.3870398,
    1.3065630,
    1.1758756,
    1.0000000,
    0.7856950,
    0.5411961,
    0.2758994,
    0.0000000,
    -0.2758994,
    -0.5411961,
    -0.7856950,
    -1.0000000,
    -1.1758756,
    -1.3065630,
    -1.3870398,
    -1.4142135,
    -1.3870398,
    -1.3065630,
    -1.1758756,
    -1.0000000,
    -0.7856950,
    -0.5411961,
    -0.2758994,
];

/// 与 `-96 dBFS` WGN 匹配的每 bin 功率下限（`GetNoiseFloorFactor`）。
fn noise_floor_factor(noise_floor_dbfs: f32) -> f32 {
    const K_DBFS_NORMALIZATION: f32 = 90.30899869919436; // 20·log10(32768)
    64.0 * 10.0f32.powf((K_DBFS_NORMALIZATION + noise_floor_dbfs) * 0.1)
}

/// 舒适噪声发生器（`ComfortNoiseGenerator`），单声道。
pub struct ComfortNoiseGenerator {
    seed: u32,
    noise_floor: f32,
    /// 启动期（前 1000 块）更慢的 N2 跟踪。
    n2_initial: Option<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    y2_smoothed: [f32; FFT_LENGTH_BY_2_PLUS_1],
    n2: [f32; FFT_LENGTH_BY_2_PLUS_1],
    n2_counter: i32,
    // 复用缓冲
    re_indices: [usize; FFT_LENGTH_BY_2 - 1],
    im_indices: [usize; FFT_LENGTH_BY_2 - 1],
}

impl Default for ComfortNoiseGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ComfortNoiseGenerator {
    pub fn new() -> Self {
        Self {
            seed: CNG_SEED,
            noise_floor: noise_floor_factor(COMFORT_NOISE_FLOOR_DBFS),
            n2_initial: Some([0.0; FFT_LENGTH_BY_2_PLUS_1]),
            y2_smoothed: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            n2: [CNG_N2_INITIAL; FFT_LENGTH_BY_2_PLUS_1],
            n2_counter: 0,
            re_indices: [0; FFT_LENGTH_BY_2 - 1],
            im_indices: [0; FFT_LENGTH_BY_2 - 1],
        }
    }

    /// 供抑制器当 masker 的噪声谱（`NoiseSpectrum`，不含 initial/floor 逻辑）。
    pub fn noise_spectrum(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.n2
    }

    /// 每块更新（`ComfortNoiseGenerator::Compute`）。
    ///
    /// `capture_spectrum` 为 nearend 谱（UsableLinearEstimate ? E2 : Y2）。
    pub fn compute(
        &mut self,
        saturated_capture: bool,
        capture_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        lower_band_noise: &mut FftData,
        upper_band_noise: &mut FftData,
    ) {
        let y2 = capture_spectrum;
        if !saturated_capture {
            // 平滑 Y2。
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.y2_smoothed[k] += CNG_Y2_SMOOTHING * (y2[k] - self.y2_smoothed[k]);
            }
            if self.n2_counter > CNG_N2_COUNTER_THRESHOLD {
                // N2 min 跟踪 + 上漂。
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.n2[k] = if self.y2_smoothed[k] < self.n2[k] {
                        (CNG_N2_MIN_UPDATE * self.y2_smoothed[k]
                            + (1.0 - CNG_N2_MIN_UPDATE) * self.n2[k])
                            * CNG_N2_UPWARD_DRIFT
                    } else {
                        self.n2[k] * CNG_N2_UPWARD_DRIFT
                    };
                }
            }
            if self.n2_initial.is_some() {
                self.n2_counter += 1;
                if self.n2_counter == CNG_N2_INITIAL_BLOCKS {
                    self.n2_initial = None;
                } else {
                    let init = self.n2_initial.as_mut().unwrap();
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        init[k] = if self.n2[k] > init[k] {
                            init[k] + CNG_N2_INITIAL_ALPHA * (self.n2[k] - init[k])
                        } else {
                            self.n2[k]
                        };
                    }
                }
            }
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.n2[k] = self.n2[k].max(self.noise_floor);
            }
            if let Some(init) = self.n2_initial.as_mut() {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    init[k] = init[k].max(self.noise_floor);
                }
            }
        }

        // 选择使用的 N2（initial 期用 initial）。
        let n2_used: [f32; FFT_LENGTH_BY_2_PLUS_1] = match self.n2_initial.as_ref() {
            Some(init) => *init,
            None => self.n2,
        };

        self.generate_random_sin_table_indices();
        Self::generate_comfort_noise(
            &n2_used,
            &self.re_indices,
            &self.im_indices,
            lower_band_noise,
            upper_band_noise,
        );
    }

    /// LCG 随机相位索引（`GenerateRandomSinTableIndices`）。
    fn generate_random_sin_table_indices(&mut self) {
        const SEED_MASK: u32 = 0x8000_0000 - 1;
        const IM_INDEX_MASK: usize = 32 - 1;
        for k in 0..(FFT_LENGTH_BY_2 - 1) {
            self.seed = self
                .seed
                .wrapping_mul(CNG_LCG_MULTIPLIER)
                .wrapping_add(1)
                & SEED_MASK;
            let index = (self.seed >> 26) as usize;
            self.re_indices[k] = index;
            self.im_indices[k] = (index + 8) & IM_INDEX_MASK;
        }
    }

    /// 噪声谱域生成（`GenerateComfortNoise`）。
    fn generate_comfort_noise(
        n2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        re_idx: &[usize; FFT_LENGTH_BY_2 - 1],
        im_idx: &[usize; FFT_LENGTH_BY_2 - 1],
        lower_band_noise: &mut FftData,
        upper_band_noise: &mut FftData,
    ) {
        let mut n = *n2;
        for v in n.iter_mut() {
            *v = v.sqrt();
        }
        // 上半频段（bin 32..=64，共 33 个）平均 → 高带平谱电平。
        let high_band_noise_level: f32 =
            n[FFT_LENGTH_BY_2_PLUS_1 / 2..].iter().sum::<f32>() / 33.0;

        lower_band_noise.re[0] = 0.0;
        lower_band_noise.re[FFT_LENGTH_BY_2] = 0.0;
        upper_band_noise.re[0] = 0.0;
        upper_band_noise.re[FFT_LENGTH_BY_2] = 0.0;
        for k in 1..FFT_LENGTH_BY_2 {
            // y = √2·sin(a)，x = √2·cos(a) = √2·sin(a+π/2)
            let x = SQRT2_SIN[re_idx[k - 1]];
            let y = SQRT2_SIN[im_idx[k - 1]];
            lower_band_noise.re[k] = n[k] * x;
            lower_band_noise.im[k] = n[k] * y;
            upper_band_noise.re[k] = high_band_noise_level * x;
            upper_band_noise.im[k] = high_band_noise_level * y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG 序列确定性 + 索引范围。
    #[test]
    fn lcg_indices_deterministic() {
        let mut cng = ComfortNoiseGenerator::new();
        cng.generate_random_sin_table_indices();
        let re1 = cng.re_indices;
        let im1 = cng.im_indices;
        cng.generate_random_sin_table_indices();
        assert_ne!(re1, cng.re_indices, "种子应推进");
        for (re, im) in re1.iter().zip(im1.iter()) {
            assert!(*re < 32);
            assert!(*im < 32);
            assert_eq!((*im as i32 - *re as i32).rem_euclid(32), 8, "虚部索引 = 实部+8 mod 32");
        }
    }

    /// N2 min 跟踪：恒定低 Y2 下 N2 收敛到噪声电平附近；上漂存在。
    #[test]
    fn n2_min_tracking() {
        let mut cng = ComfortNoiseGenerator::new();
        // 恒定 Y2 = 1000（每 bin）
        let y2 = [1000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut low = FftData::default();
        let mut high = FftData::default();
        for _ in 0..2000 {
            cng.compute(false, &y2, &mut low, &mut high);
        }
        let n2 = cng.noise_spectrum();
        // initial 期结束（1000 块），N2 应接近 1000（min 跟踪 + 漂移平衡）
        for k in [1usize, 16, 32, 64] {
            assert!(
                n2[k] > 500.0 && n2[k] < 1600.0,
                "k={} N2={}",
                k,
                n2[k]
            );
        }
        // Y2 突降 → N2 快速下追
        let y2_low = [10.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for _ in 0..100 {
            cng.compute(false, &y2_low, &mut low, &mut high);
        }
        assert!(cng.noise_spectrum()[16] < 500.0);
    }

    /// 饱和时冻结：N2 不变、噪声仍生成。
    #[test]
    fn saturated_freezes_estimate() {
        let mut cng = ComfortNoiseGenerator::new();
        let y2 = [1000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut low = FftData::default();
        let mut high = FftData::default();
        for _ in 0..100 {
            cng.compute(false, &y2, &mut low, &mut high);
        }
        let frozen = *cng.noise_spectrum();
        for _ in 0..50 {
            cng.compute(true, &[1e9; FFT_LENGTH_BY_2_PLUS_1], &mut low, &mut high);
        }
        assert_eq!(frozen, *cng.noise_spectrum());
    }

    /// 噪声谱基本性质：DC/Nyquist 为零、低带按 N2 成形、确定性。
    #[test]
    fn noise_shape() {
        let mut cng = ComfortNoiseGenerator::new();
        let y2 = [40000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut low = FftData::default();
        let mut high = FftData::default();
        for _ in 0..1500 {
            cng.compute(false, &y2, &mut low, &mut high);
        }
        assert_eq!(low.re[0], 0.0);
        assert_eq!(low.re[FFT_LENGTH_BY_2], 0.0);
        // |N[k]|² ≈ N2[k]（√2·sin 与 √2·cos 的平方和为 2，期望幅度²≈N2）
        // 单次实现的方差大，只验证数量级
        let mag2 = low.re[16] * low.re[16] + low.im[16] * low.im[16];
        assert!(mag2 > 1000.0 && mag2 < 200000.0, "mag2={}", mag2);
        // 确定性：重放同状态得到同输出
        let low2 = low;
        let mut cng2 = ComfortNoiseGenerator::new();
        let mut l2 = FftData::default();
        let mut h2 = FftData::default();
        for _ in 0..1500 {
            cng2.compute(false, &y2, &mut l2, &mut h2);
        }
        assert_eq!(low2.re[16], l2.re[16]);
    }
}
