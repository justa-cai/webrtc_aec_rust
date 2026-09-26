//! 混响尾部模型（简化），对照 `modules/audio_processing/aec3/reverb_model.cc`。
//!
//! 一阶泄漏积分器（功率域指数衰减卷积）：
//! `reverb[k] = (reverb[k] + power[k]·scaling) · decay`。
//!
//! 简化（见 README 偏差清单）：
//! - decay 恒 0.83（`ep_strength.default_len ≥ 0` 时上游亦不自适应）；
//! - 线性模式的频响 scaling 用 `max(tail[k], direct[k]·decay)` 现算，
//!   不做上游 ReverbFrequencyResponse 的 ERLE 质量平滑与平稳门控。

use crate::constants::{EP_STRENGTH_DEFAULT_LEN, FFT_LENGTH_BY_2_PLUS_1};

/// 混响功率模型（`ReverbModel`）。
#[derive(Clone, Debug)]
pub struct ReverbModel {
    reverb: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for ReverbModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ReverbModel {
    pub fn new() -> Self {
        Self {
            reverb: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    pub fn reset(&mut self) {
        self.reverb = [0.0; FFT_LENGTH_BY_2_PLUS_1];
    }

    /// 线性模式更新（逐 bin scaling，`UpdateReverb`）。
    pub fn update_reverb(
        &mut self,
        power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        scaling: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        decay: f32,
    ) {
        if decay > 0.0 {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.reverb[k] = (self.reverb[k] + power[k] * scaling[k]) * decay;
            }
        }
    }

    /// 非线性模式更新（标量 scaling，`UpdateReverbNoFreqShaping`）。
    pub fn update_reverb_no_freq_shaping(
        &mut self,
        power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        scaling: f32,
        decay: f32,
    ) {
        if decay > 0.0 {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.reverb[k] = (self.reverb[k] + power[k] * scaling) * decay;
            }
        }
    }

    /// 把混响功率加到 R2（`AddReverb`）。
    pub fn add_reverb(&self, r2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            r2[k] += self.reverb[k];
        }
    }
}

/// 默认混响衰减（恒定，`ep_strength.default_len`）。
pub const REVERB_DECAY: f32 = EP_STRENGTH_DEFAULT_LEN;

/// 简化的混响频响：`max(tail[k], direct[k]·avg_decay)`。
///
/// `avg_decay = Σ_{k≥1} tail[k] / Σ_{k≥1} direct[k]`（ReverbFrequencyResponse
/// 的 AverageDecayWithinFilter，去 DC 的能量比）——量级 ≈ 尾频响本身，
/// **不是** direct·0.83（那会放大 1~2 个数量级）。保守 max(实际尾) 与源码一致。
///
/// `h2` 为精滤波器各分区功率频响（H2[p]），`delay_blocks` 为直达路径分区。
pub fn simplified_reverb_frequency_response(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    delay_blocks: usize,
) -> [f32; FFT_LENGTH_BY_2_PLUS_1] {
    let mut out = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
    if h2.is_empty() {
        return out;
    }
    let direct = &h2[delay_blocks.min(h2.len() - 1)];
    let tail = &h2[h2.len() - 1];
    let sum_tail: f32 = tail.iter().skip(1).sum();
    let sum_direct: f32 = direct.iter().skip(1).sum();
    let avg_decay = if sum_direct > 0.0 {
        sum_tail / sum_direct
    } else {
        0.0
    };
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        out[k] = tail[k].max(direct[k] * avg_decay);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 泄漏积分器：恒定输入下收敛到 power·scaling·decay/(1−decay)。
    #[test]
    fn leaky_integrator_converges() {
        let mut m = ReverbModel::new();
        let power = [1000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for _ in 0..1000 {
            m.update_reverb_no_freq_shaping(&power, 1.0, 0.83);
        }
        let expect = 1000.0 * 0.83 / (1.0 - 0.83);
        assert!((m.reverb[16] - expect).abs() < 5.0, "reverb={}", m.reverb[16]);
    }

    /// 简化频响：max(tail, direct·avg_decay)，avg_decay=Σtail/Σdirect。
    #[test]
    fn simplified_freq_response() {
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 3];
        h2[0][16] = 10.0; // direct（delay=0）
        h2[2][16] = 3.0; // tail
        let r = simplified_reverb_frequency_response(&h2, 0);
        // avg_decay = 3/10 → direct·avg_decay = 3 → max(3, 3) = 3（尾量级，非 direct·0.83=8.3）
        assert!((r[16] - 3.0).abs() < 1e-5, "r={}", r[16]);
        // 尾更大时取尾
        h2[2][16] = 20.0;
        let r2 = simplified_reverb_frequency_response(&h2, 0);
        assert!((r2[16] - 20.0).abs() < 1e-6);
    }
}
