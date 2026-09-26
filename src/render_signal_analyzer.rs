//! render 信号分析（窄带检测），对照
//! `modules/audio_processing/aec3/render_signal_analyzer.{h,cc}`。
//!
//! 只保留线性路径用到的部分：
//! - 窄带计数器（`IdentifySmallNarrowBandRegions`）：在对齐位置的功率谱上，
//!   bin 比两侧邻居高 3 倍即视为命中，命中递增、未命中清零；
//! - `PoorSignalExcitation`：任一计数器 > 10（激励不足判定）；
//! - `MaskRegionsAroundNarrowBands`：计数器 > 5 时把 5-bin 邻域置零。
//!
//! 未移植：`IdentifyStrongNarrowBandComponent`（上游仅供抑制器与数据导出使用）。

use crate::constants::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::render_delay_buffer::RenderBufferView;

/// 窄带计数门限（`kCounterThreshold`）。
const COUNTER_THRESHOLD: usize = 5;
/// PoorSignalExcitation 门限：计数器持续超过该值视为激励不足。
const POOR_EXCITATION_COUNTER: usize = 10;

/// render 信号分析器（`RenderSignalAnalyzer`）。
#[derive(Clone, Debug)]
pub struct RenderSignalAnalyzer {
    /// 每 bin（k=1..63 存于 k−1）的连续窄带命中计数。
    narrow_band_counters: [usize; FFT_LENGTH_BY_2 - 1],
}

impl Default for RenderSignalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderSignalAnalyzer {
    pub fn new() -> Self {
        Self {
            narrow_band_counters: [0; FFT_LENGTH_BY_2 - 1],
        }
    }

    /// 每块更新（`RenderSignalAnalyzer::Update` 的窄带部分）。
    ///
    /// `delay_partitions` 为对齐的直达路径滤波器延迟（None 时清零计数器）。
    pub fn update(&mut self, render_buffer: &RenderBufferView, delay_partitions: Option<usize>) {
        let Some(d) = delay_partitions else {
            self.narrow_band_counters = [0; FFT_LENGTH_BY_2 - 1];
            return;
        };
        let x2 = render_buffer.spectrum(d as isize);
        for k in 1..FFT_LENGTH_BY_2 {
            let hit = x2[k] > 3.0 * x2[k - 1].max(x2[k + 1]);
            let idx = k - 1;
            self.narrow_band_counters[idx] = if hit {
                self.narrow_band_counters[idx] + 1
            } else {
                0
            };
        }
    }

    /// 激励是否不足（`PoorSignalExcitation`）。
    pub fn poor_signal_excitation(&self) -> bool {
        self.narrow_band_counters
            .iter()
            .any(|c| *c > POOR_EXCITATION_COUNTER)
    }

    /// 把窄带区域邻域（5 bin）置零（`MaskRegionsAroundNarrowBands`）。
    pub fn mask_regions_around_narrow_bands(&self, v: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        if self.narrow_band_counters[0] > COUNTER_THRESHOLD {
            v[0] = 0.0;
            v[1] = 0.0;
        }
        for k in 2..(FFT_LENGTH_BY_2 - 1) {
            if self.narrow_band_counters[k - 1] > COUNTER_THRESHOLD {
                for j in (k - 2)..=(k + 2) {
                    v[j] = 0.0;
                }
            }
        }
        if self.narrow_band_counters[FFT_LENGTH_BY_2 - 2] > COUNTER_THRESHOLD {
            v[FFT_LENGTH_BY_2 - 1] = 0.0;
            v[FFT_LENGTH_BY_2] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T22（单元部分）：2 kHz 正弦 → 窄带命中、掩蔽 5-bin 邻域。
    #[test]
    fn narrow_band_detection_and_masking() {
        let mut a = RenderSignalAnalyzer::new();
        // 构造带窄峰的谱（bin 16 处尖峰）+ 手动喂计数器逻辑：
        // 直接调用 update 需要完整 RenderBufferView；此处用内部计数器验证掩蔽。
        // （窄带命中逻辑由 M3 集成测试覆盖完整链路。）
        a.narrow_band_counters[16 - 1] = 6; // bin 16 命中 6 次
        assert!(a.poor_signal_excitation() == false); // 6 ≤ 10 尚不算 poor
        a.narrow_band_counters[16 - 1] = 11;
        assert!(a.poor_signal_excitation());
        let mut v = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        a.mask_regions_around_narrow_bands(&mut v);
        // bin 16 命中 → 清零 v[14..=18]（k−2..=k+2）
        for j in 14..=18 {
            assert_eq!(v[j], 0.0, "bin {} 应被掩蔽", j);
        }
        assert_eq!(v[13], 1.0);
        assert_eq!(v[19], 1.0);
        // 边界：bin 1 命中 → v[0..=1] 清零
        let mut b = RenderSignalAnalyzer::new();
        b.narrow_band_counters[0] = 6;
        let mut v2 = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        b.mask_regions_around_narrow_bands(&mut v2);
        assert_eq!(v2[0], 0.0);
        assert_eq!(v2[1], 0.0);
        assert_eq!(v2[2], 1.0);
    }
}
