//! 频域数据容器，对照 `modules/audio_processing/aec3/fft_data.h`。
//!
//! 128 点实数 FFT 的前 65 个 bin（DC..Nyquist）。
//! `im[0]` 与 `im[64]`（DC 与 Nyquist）恒为 0。

use crate::constants::FFT_LENGTH_BY_2_PLUS_1;

/// 128 点实值 FFT 产生的半谱数据（`FftData`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FftData {
    pub re: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub im: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for FftData {
    fn default() -> Self {
        Self {
            re: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            im: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }
}

impl FftData {
    /// 清零（`FftData::Clear`）。
    pub fn clear(&mut self) {
        self.re.fill(0.0);
        self.im.fill(0.0);
    }

    /// 拷贝并强制 DC/Nyquist 虚部为 0（`FftData::Assign`）。
    pub fn assign(&mut self, src: &FftData) {
        self.re.copy_from_slice(&src.re);
        self.im.copy_from_slice(&src.im);
        self.im[0] = 0.0;
        self.im[FFT_LENGTH_BY_2_PLUS_1 - 1] = 0.0;
    }

    /// 未归一化功率谱（`FftData::Spectrum`）：`out[k] = re[k]² + im[k]²`。
    pub fn spectrum(&self, out: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            out[k] = self.re[k] * self.re[k] + self.im[k] * self.im[k];
        }
    }
}
