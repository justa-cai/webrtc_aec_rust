//! 4 倍降采样器，对照 `modules/audio_processing/aec3/decimator.cc` 与
//! `modules/audio_processing/utility/cascaded_biquad_filter.cc`。
//!
//! 链路：3 节椭圆低通（`signal.ellip(6, 1, 40, 1800/8000, 'lowpass')`）
//! → 1 节 Butterworth 高通（`signal.butter(2, 1000/8000, 'highpass')`）
//! → 每 4 个样本取 1 个。
//!
//! biquad 为**直接 I 型**（b[3]、a[2]，跨调用保持状态）。

use crate::constants::BLOCK_SIZE;

/// 单节 biquad 系数：H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a0·z⁻¹ + a1·z⁻²)。
#[derive(Clone, Copy, Debug, Default)]
pub struct BiQuadCoefficients {
    pub b: [f32; 3],
    pub a: [f32; 2],
}

/// 直接 I 型 biquad，带输入/输出历史（`cascaded_biquad_filter.h` 的 `BiQuadFilter`）。
#[derive(Clone, Debug, Default)]
pub struct BiQuad {
    coefficients: BiQuadCoefficients,
    x: [f32; 2], // 输入历史 x[n-1], x[n-2]
    y: [f32; 2], // 输出历史 y[n-1], y[n-2]
}

impl BiQuad {
    pub fn new(coefficients: BiQuadCoefficients) -> Self {
        Self {
            coefficients,
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    /// 单样本递推：`y[k] = b0·x + b1·x0 + b2·x1 − a0·y0 − a1·y1`（`ApplyBiQuad`）。
    pub fn process(&mut self, input: f32) -> f32 {
        let c = &self.coefficients;
        let output = c.b[0] * input
            + c.b[1] * self.x[0]
            + c.b[2] * self.x[1]
            - c.a[0] * self.y[0]
            - c.a[1] * self.y[1];
        self.x[1] = self.x[0];
        self.x[0] = input;
        self.y[1] = self.y[0];
        self.y[0] = output;
        output
    }

    pub fn reset(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// 级联 biquad（`CascadedBiQuadFilter`）：各节顺序处理，前节输出为后节输入。
#[derive(Clone, Debug)]
pub struct CascadedBiQuad {
    biquads: Vec<BiQuad>,
}

impl CascadedBiQuad {
    pub fn new(coefficients: &[BiQuadCoefficients]) -> Self {
        Self {
            biquads: coefficients.iter().map(|c| BiQuad::new(*c)).collect(),
        }
    }

    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        if self.biquads.is_empty() {
            y.copy_from_slice(x);
            return;
        }
        // 逐节顺序处理：第 j 节的输出作为第 j+1 节的输入（对应 C++ 的 x→y 再 y→y 原地）。
        let mut cur = x.to_vec();
        for bq in self.biquads.iter_mut() {
            for v in cur.iter_mut() {
                *v = bq.process(*v);
            }
        }
        y.copy_from_slice(&cur);
    }

    pub fn reset(&mut self) {
        for bq in self.biquads.iter_mut() {
            bq.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// ds4 系数 —— 从 decimator.cc 原样照抄
// ---------------------------------------------------------------------------

/// 3 节椭圆低通（1800 Hz 截止，40 dB 阻带）。
const LOW_PASS_FILTER_DS4: [BiQuadCoefficients; 3] = [
    BiQuadCoefficients {
        b: [0.0180919877, 0.00320961363, 0.0180919877],
        a: [-1.5183195, 0.633165865],
    },
    BiQuadCoefficients {
        b: [1.0, -1.24550459, 1.0],
        a: [-1.49784254, 0.853586692],
    },
    BiQuadCoefficients {
        b: [1.0, -1.4221681, 1.0],
        a: [-1.49791282, 0.969572384],
    },
];

/// 1 节 Butterworth 高通（1000 Hz），抑制低频近端噪声。
const HIGH_PASS_FILTER: [BiQuadCoefficients; 1] = [BiQuadCoefficients {
    b: [0.757076375, -1.51415275, 0.757076375],
    a: [-1.45424359, 0.574061915],
}];

/// 4 倍（或 8 倍）降采样器（`Decimator`）。
///
/// 8 倍路径（带通）未移植：本实现固定 `down_sampling_factor = 4`。
pub struct Decimator {
    down_sampling_factor: usize,
    anti_aliasing_filter: CascadedBiQuad,
    noise_reduction_filter: CascadedBiQuad,
    /// 复用缓冲（抗混叠输出 / 高通输出）
    tmp1: Vec<f32>,
    tmp2: Vec<f32>,
}

impl Decimator {
    pub fn new(down_sampling_factor: usize) -> Self {
        assert!(down_sampling_factor == 4, "仅移植 ds4 路径");
        Self {
            down_sampling_factor,
            anti_aliasing_filter: CascadedBiQuad::new(&LOW_PASS_FILTER_DS4),
            noise_reduction_filter: CascadedBiQuad::new(&HIGH_PASS_FILTER),
            tmp1: vec![0.0; BLOCK_SIZE],
            tmp2: vec![0.0; BLOCK_SIZE],
        }
    }

    /// 64 样本 → 16 样本（`Decimator::Decimate`）：抗混叠 → 高通 → 抽取。
    pub fn decimate(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), BLOCK_SIZE);
        assert_eq!(output.len(), BLOCK_SIZE / self.down_sampling_factor);
        self.anti_aliasing_filter.process(input, &mut self.tmp1);
        self.noise_reduction_filter.process(&self.tmp1, &mut self.tmp2);
        for (j, out) in output.iter_mut().enumerate() {
            *out = self.tmp2[j * self.down_sampling_factor];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq_hz: f32, n: usize, phase: &mut f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push((*phase).sin() * 1000.0);
            *phase += 2.0 * std::f32::consts::PI * freq_hz / 16000.0;
        }
        v
    }

    /// f64 参考 biquad 级联（直接 I 型），用于稳态增益校验。
    fn reference_cascade(coeffs: &[BiQuadCoefficients], input: &[f32]) -> Vec<f64> {
        let mut out: Vec<f64> = input.iter().map(|v| *v as f64).collect();
        for c in coeffs {
            let (mut xm1, mut xm2, mut ym1, mut ym2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for v in out.iter_mut() {
                let inp = *v;
                let o = c.b[0] as f64 * inp
                    + c.b[1] as f64 * xm1
                    + c.b[2] as f64 * xm2
                    - c.a[0] as f64 * ym1
                    - c.a[1] as f64 * ym2;
                xm2 = xm1;
                xm1 = inp;
                ym2 = ym1;
                ym1 = o;
                *v = o;
            }
        }
        out
    }

    /// T4a: 1400 Hz（通带内）稳态增益匹配 f64 级联参考。
    #[test]
    fn decimator_passband_gain() {
        let mut dec = Decimator::new(4);
        let mut phase = 0.0f32;
        // 先冲 200 块让滤波器进入稳态
        for _ in 0..200 {
            let x = sine(1400.0, BLOCK_SIZE, &mut phase);
            let mut out = [0.0f32; BLOCK_SIZE / 4];
            dec.decimate(&x, &mut out);
        }
        // 参考级联（同样相位连续）
        let mut phase_ref = 0.0f32;
        let mut big_in = Vec::new();
        for _ in 0..201 {
            big_in.extend(sine(1400.0, BLOCK_SIZE, &mut phase_ref));
        }
        let all_coeffs = LOW_PASS_FILTER_DS4
            .iter()
            .chain(HIGH_PASS_FILTER.iter())
            .copied()
            .collect::<Vec<_>>();
        let ref_out = reference_cascade(&all_coeffs, &big_in);

        let x = sine(1400.0, BLOCK_SIZE, &mut phase);
        let mut out = [0.0f32; BLOCK_SIZE / 4];
        dec.decimate(&x, &mut out);
        for j in 0..16 {
            let idx = 200 * BLOCK_SIZE + j * 4;
            assert!(
                (out[j] as f64 - ref_out[idx]).abs() < 1e-2,
                "j={}: {} vs {}",
                j,
                out[j],
                ref_out[idx]
            );
        }
    }

    /// T4b: 5 kHz（阻带）衰减 ≥ 30 dB。
    #[test]
    fn decimator_stopband_attenuation() {
        let mut dec = Decimator::new(4);
        let mut phase = 0.0f32;
        let mut in_rms = 0.0f64;
        let mut out_rms = 0.0f64;
        for _ in 0..300 {
            let x = sine(5000.0, BLOCK_SIZE, &mut phase);
            in_rms += (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()) / 300.0;
            let mut out = [0.0f32; BLOCK_SIZE / 4];
            dec.decimate(&x, &mut out);
            out_rms += (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() * 4.0) / 300.0;
        }
        let atten_db = 10.0 * (in_rms / out_rms.max(1e-20)).log10();
        assert!(
            atten_db > 30.0,
            "5 kHz 衰减不足: {} dB (in_rms={} out_rms={})",
            atten_db,
            in_rms,
            out_rms
        );
    }

    /// T4c: 抽取位置正确（out[j] == 滤波后 x[4j]）。
    #[test]
    fn decimator_decimation_positions() {
        let mut dec = Decimator::new(4);
        let x: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32 * 0.1).sin() * 500.0).collect();
        let all_coeffs = LOW_PASS_FILTER_DS4
            .iter()
            .chain(HIGH_PASS_FILTER.iter())
            .copied()
            .collect::<Vec<_>>();
        let ref_out = reference_cascade(&all_coeffs, &x);
        let mut out = [0.0f32; BLOCK_SIZE / 4];
        dec.decimate(&x, &mut out);
        for j in 0..16 {
            assert!(
                (out[j] as f64 - ref_out[j * 4]).abs() < 1e-2,
                "j={}: {} vs {}",
                j,
                out[j],
                ref_out[j * 4]
            );
        }
    }

    /// T4d: 长时间稳定性（无 NaN/Inf）。
    #[test]
    fn decimator_stability() {
        let mut dec = Decimator::new(4);
        let mut state = 0x12345678u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) as f32 / (1u64 << 30) as f32 - 1.0) * 3000.0
        };
        for _ in 0..(1_000_000 / BLOCK_SIZE) {
            let x: Vec<f32> = (0..BLOCK_SIZE).map(|_| next()).collect();
            let mut out = [0.0f32; BLOCK_SIZE / 4];
            dec.decimate(&x, &mut out);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
