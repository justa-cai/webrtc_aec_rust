//! 128 点实数 FFT 封装，对照 `modules/audio_processing/aec3/aec3_fft.{h,cc}`。
//!
//! # 缩放约定（与 WebRTC 的等价性推导）
//!
//! WebRTC 的 Ooura 封装：前向不缩放，`InverseFft(Fft(x)) = 64·x`，各调用点自行乘
//! `1/64`（`PredictionError` 的 `kScale`、`Constrain` 的 `kScale`）。
//! 本实现改用自洽约定：**前向不缩放、逆向乘 1/128**，往返严格相等
//! （`ifft(fft(x)) == x`）。因此：
//! - 预测误差：`e = y − ifft(S)[64..128]`，无额外因子；
//! - 时域约束：`ifft → 清零 [64..128) → fft`，无额外因子。
//! 两个约定下 X/H/G/S/E/Y 数值逐数组一致。
//!
//! 实数 FFT 通过复数 FFT 实现：实序列填零虚部做 128 点复数 FFT，取 bin 0..=64
//! （共轭对称）；逆变换时把 bin 65..127 镜像为共轭后取实部。

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::constants::{FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::fft_data::FftData;

// ---------------------------------------------------------------------------
// 窗表 —— 从 aec3_fft.cc 原样照抄，不要重新生成
// ---------------------------------------------------------------------------

/// 64 点 Hann 窗（`kHanning64`，用于误差块的 ZeroPaddedFft）。
const HANNING_64: [f32; FFT_LENGTH_BY_2] = [
    0.0,
    0.00248461,
    0.00991376,
    0.0222136,
    0.03926189,
    0.06088921,
    0.08688061,
    0.11697778,
    0.15088159,
    0.1882551,
    0.22872687,
    0.27189467,
    0.31732949,
    0.36457977,
    0.41317591,
    0.46263495,
    0.51246535,
    0.56217185,
    0.61126047,
    0.65924333,
    0.70564355,
    0.75,
    0.79187184,
    0.83084292,
    0.86652594,
    0.89856625,
    0.92664544,
    0.95048443,
    0.96984631,
    0.98453864,
    0.99441541,
    0.99937846,
    0.99937846,
    0.99441541,
    0.98453864,
    0.96984631,
    0.95048443,
    0.92664544,
    0.89856625,
    0.86652594,
    0.83084292,
    0.79187184,
    0.75,
    0.70564355,
    0.65924333,
    0.61126047,
    0.56217185,
    0.51246535,
    0.46263495,
    0.41317591,
    0.36457977,
    0.31732949,
    0.27189467,
    0.22872687,
    0.1882551,
    0.15088159,
    0.11697778,
    0.08688061,
    0.06088921,
    0.03926189,
    0.0222136,
    0.00991376,
    0.00248461,
    0.0,
];

/// 128 点 sqrt-Hann 窗（`kSqrtHanning128`，用于 Y/E 的 WindowedPaddedFft）。
/// Matlab: `win = sqrt(hanning(128))`。
const SQRT_HANNING_128: [f32; FFT_LENGTH] = [
    0.00000000000000,
    0.02454122852291,
    0.04906767432742,
    0.07356456359967,
    0.09801714032956,
    0.12241067519922,
    0.14673047445536,
    0.17096188876030,
    0.19509032231613,
    0.21910124015687,
    0.24298017990326,
    0.26671275747490,
    0.29028467725446,
    0.31368174039889,
    0.33688985334222,
    0.35989503653499,
    0.38268343236528,
    0.40524131400499,
    0.42755509343028,
    0.44961132965461,
    0.47139673682600,
    0.49289819222978,
    0.51410274419322,
    0.53499761988710,
    0.55557023301960,
    0.57580831714507,
    0.59569930449243,
    0.61523159058063,
    0.63439328416365,
    0.65317284295378,
    0.67155895484702,
    0.68954054473707,
    0.70710678118655,
    0.72424713763143,
    0.74095112535496,
    0.75720884650648,
    0.77301045336274,
    0.78834642762661,
    0.80327157145752,
    0.81758481315158,
    0.83146961230255,
    0.84485356524971,
    0.85772861000027,
    0.87008699110871,
    0.88192126434835,
    0.89322430119552,
    0.90398929312344,
    0.91420975570353,
    0.92387953251129,
    0.93299279883474,
    0.94154406518302,
    0.94952818059304,
    0.95694033572221,
    0.96377606579544,
    0.97003125319454,
    0.97570213003853,
    0.98078528040323,
    0.98527764238894,
    0.98917650996478,
    0.99247953459871,
    0.99518472667220,
    0.99729045667869,
    0.99879545620517,
    0.99969881869620,
    1.00000000000000,
    0.99969881869620,
    0.99879545620517,
    0.99729045667869,
    0.99518472667220,
    0.99247953459871,
    0.98917650996478,
    0.98527764238894,
    0.98078528040323,
    0.97570213003853,
    0.97003125319454,
    0.96377606579544,
    0.95694033572221,
    0.94952818059304,
    0.94154406518302,
    0.93299279883474,
    0.92387953251129,
    0.91420975570353,
    0.90398929312344,
    0.89322430119552,
    0.88192126434835,
    0.87008699110871,
    0.85772861000027,
    0.84485356524971,
    0.83146961230255,
    0.81758481315158,
    0.80327157145752,
    0.78834642762661,
    0.77301045336274,
    0.75720884650648,
    0.74095112535496,
    0.72424713763143,
    0.70710678118655,
    0.68954054473707,
    0.67155895484702,
    0.65317284295378,
    0.63439328416365,
    0.61523159058063,
    0.59569930449243,
    0.57580831714507,
    0.55557023301960,
    0.53499761988710,
    0.51410274419322,
    0.49289819222978,
    0.47139673682600,
    0.44961132965461,
    0.42755509343028,
    0.40524131400499,
    0.38268343236528,
    0.35989503653499,
    0.33688985334222,
    0.31368174039889,
    0.29028467725446,
    0.26671275747490,
    0.24298017990326,
    0.21910124015687,
    0.19509032231613,
    0.17096188876030,
    0.14673047445536,
    0.12241067519922,
    0.09801714032956,
    0.07356456359967,
    0.04906767432742,
    0.02454122852291,
];

/// 访问 √Hann128 窗表（供 SuppressionFilter 的 WOLA 合成使用）。
pub fn sqrt_hanning(i: usize) -> f32 {
    SQRT_HANNING_128[i]
}

/// 时域窗类型（`Aec3Fft::Window`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// 矩形窗：PaddedFft 拼接 [x_old, x] 直接变换。
    Rectangular,
    /// 64 点 Hann：ZeroPaddedFft 对误差块加窗。
    Hanning,
    /// 128 点 sqrt-Hann：WindowedPaddedFft 对 [x_old, x] 加窗。
    SqrtHanning,
}

/// 提供 128 点实值 FFT 功能（`Aec3Fft`）。
///
/// 方法取 `&mut self` 以复用内部 scratch/复数缓冲，避免每次调用分配。
pub struct Aec3Fft {
    forward: std::sync::Arc<dyn Fft<f32>>,
    inverse: std::sync::Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    buf: Vec<Complex<f32>>,
}

impl Default for Aec3Fft {
    fn default() -> Self {
        Self::new()
    }
}

impl Aec3Fft {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_LENGTH);
        let inverse = planner.plan_fft_inverse(FFT_LENGTH);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        Self {
            forward,
            inverse,
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            buf: vec![Complex::new(0.0, 0.0); FFT_LENGTH],
        }
    }

    /// 正变换：128 点实序列 → 半谱。前向不缩放（`Aec3Fft::Fft`）。
    pub fn fft(&mut self, x_real: &[f32; FFT_LENGTH], out: &mut FftData) {
        for n in 0..FFT_LENGTH {
            self.buf[n] = Complex {
                re: x_real[n],
                im: 0.0,
            };
        }
        self.forward
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            out.re[k] = self.buf[k].re;
            out.im[k] = self.buf[k].im;
        }
        out.im[0] = 0.0;
        out.im[FFT_LENGTH_BY_2] = 0.0;
    }

    /// 逆变换：半谱 → 128 点实序列。逆向含 1/128 缩放，往返严格相等。
    /// 等价于 WebRTC 的 `Ifft(结果)·(1/64)`。
    pub fn ifft(&mut self, x: &FftData, out_real: &mut [f32; FFT_LENGTH]) {
        self.buf[0] = Complex {
            re: x.re[0],
            im: 0.0,
        };
        self.buf[FFT_LENGTH_BY_2] = Complex {
            re: x.re[FFT_LENGTH_BY_2],
            im: 0.0,
        };
        for k in 1..FFT_LENGTH_BY_2 {
            self.buf[k] = Complex {
                re: x.re[k],
                im: x.im[k],
            };
            // 镜像共轭
            self.buf[FFT_LENGTH - k] = Complex {
                re: x.re[k],
                im: -x.im[k],
            };
        }
        self.inverse
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        let scale = 1.0 / FFT_LENGTH as f32;
        for n in 0..FFT_LENGTH {
            out_real[n] = self.buf[n].re * scale;
        }
    }

    /// 拼接 `[x_old, x]` 成 128 点后做 FFT（`Aec3Fft::PaddedFft`）。
    /// 矩形窗用于 render 块入库；sqrt-Hann 窗用于 Y/E 谱计算。
    pub fn padded_fft(
        &mut self,
        x: &[f32; FFT_LENGTH_BY_2],
        x_old: &[f32; FFT_LENGTH_BY_2],
        window: Window,
        out: &mut FftData,
    ) {
        let mut fft_in = [0.0f32; FFT_LENGTH];
        match window {
            Window::Rectangular => {
                fft_in[..FFT_LENGTH_BY_2].copy_from_slice(x_old);
                fft_in[FFT_LENGTH_BY_2..].copy_from_slice(x);
            }
            Window::SqrtHanning => {
                for k in 0..FFT_LENGTH_BY_2 {
                    fft_in[k] = x_old[k] * SQRT_HANNING_128[k];
                    fft_in[k + FFT_LENGTH_BY_2] = x[k] * SQRT_HANNING_128[k + FFT_LENGTH_BY_2];
                }
            }
            Window::Hanning => unreachable!("PaddedFft 不支持 Hanning 窗"),
        }
        self.fft(&fft_in, out);
    }

    /// 前半补零后对当前块做 FFT（`Aec3Fft::ZeroPaddedFft`）。
    /// Hanning 窗用于误差块 e 的谱计算。
    pub fn zero_padded_fft(
        &mut self,
        x: &[f32; FFT_LENGTH_BY_2],
        window: Window,
        out: &mut FftData,
    ) {
        let mut fft_in = [0.0f32; FFT_LENGTH];
        match window {
            Window::Rectangular => {
                fft_in[FFT_LENGTH_BY_2..].copy_from_slice(x);
            }
            Window::Hanning => {
                for k in 0..FFT_LENGTH_BY_2 {
                    fft_in[k + FFT_LENGTH_BY_2] = x[k] * HANNING_64[k];
                }
            }
            Window::SqrtHanning => unreachable!("ZeroPaddedFft 不支持 SqrtHanning 窗"),
        }
        self.fft(&fft_in, out);
    }

    /// sqrt-Hann 加窗的 PaddedFft 并更新 x_old（echo_remover.cc `WindowedPaddedFft`）。
    pub fn windowed_padded_fft(
        &mut self,
        v: &[f32; FFT_LENGTH_BY_2],
        v_old: &mut [f32; FFT_LENGTH_BY_2],
        out: &mut FftData,
    ) {
        self.padded_fft(v, v_old, Window::SqrtHanning, out);
        v_old.copy_from_slice(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机数（xorshift64*），全 crate 测试共用。
    pub struct Rng(u64);
    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        pub fn next_f32(&mut self) -> f32 {
            // [0,1)
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545F4914F6CDD1D);
            ((v >> 40) as f32) / (1u64 << 24) as f32
        }
    }

    /// f64 直接 DFT 参考（测试用）。
    fn dft_reference(x: &[f32; FFT_LENGTH], k: usize) -> (f64, f64) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (n, &v) in x.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / FFT_LENGTH as f64;
            re += v as f64 * ang.cos();
            im += v as f64 * ang.sin();
        }
        (re, im)
    }

    /// T1: FFT 往返恒等。
    #[test]
    fn fft_roundtrip() {
        let mut fft = Aec3Fft::new();
        let mut rng = Rng::new(12345);
        for _ in 0..100 {
            let mut x = [0.0f32; FFT_LENGTH];
            let mut max_abs = 0.0f32;
            for v in x.iter_mut() {
                *v = (rng.next_f32() * 2.0 - 1.0) * 3000.0;
                max_abs = max_abs.max(v.abs());
            }
            let mut X = FftData::default();
            fft.fft(&x, &mut X);
            let mut y = [0.0f32; FFT_LENGTH];
            fft.ifft(&X, &mut y);
            for n in 0..FFT_LENGTH {
                assert!(
                    (y[n] - x[n]).abs() < 2e-4 * max_abs,
                    "往返误差过大: n={} diff={} max={}",
                    n,
                    y[n] - x[n],
                    max_abs
                );
            }
        }
    }

    /// T2: bin 值对 f64 直接 DFT 校验（DC 精确、单音 bin 匹配）。
    #[test]
    fn fft_bins_vs_dft() {
        let mut fft = Aec3Fft::new();
        // DC：全 1 → X[0] = 128
        let mut X = FftData::default();
        fft.fft(&[1.0; FFT_LENGTH], &mut X);
        assert!((X.re[0] - 128.0).abs() < 1e-3);
        assert!(X.im[0].abs() < 1e-3);
        for k in 1..FFT_LENGTH_BY_2_PLUS_1 {
            assert!(X.re[k].abs() < 1e-3, "k={} re={}", k, X.re[k]);
            assert!(X.im[k].abs() < 1e-3, "k={} im={}", k, X.im[k]);
        }

        // 单音 16 cycles/128（bin 16）
        let mut x = [0.0f32; FFT_LENGTH];
        for (n, v) in x.iter_mut().enumerate() {
            *v = (2.0 * std::f32::consts::PI * 16.0 * n as f32 / FFT_LENGTH as f32).cos() * 1000.0;
        }
        fft.fft(&x, &mut X);
        let (ref_re, ref_im) = dft_reference(&x, 16);
        assert!((X.re[16] as f64 - ref_re).abs() < 1e-2 * ref_re.abs().max(1.0));
        assert!((X.im[16] as f64 - ref_im).abs() < 1e-2 * ref_im.abs().max(1.0));
        // 其他 bin 应接近 0
        for k in [0usize, 8, 32, 64] {
            let (r, i) = dft_reference(&x, k);
            assert!((X.re[k] as f64 - r).abs() < 1.0, "k={}", k);
            assert!((X.im[k] as f64 - i).abs() < 1.0, "k={}", k);
        }
    }

    /// T3: PaddedFft(矩形窗) == 对 [x_old, x] 的直接 128 点 DFT。
    #[test]
    fn padded_fft_equivalence() {
        let mut fft = Aec3Fft::new();
        let mut rng = Rng::new(777);
        let mut x_old = [0.0f32; FFT_LENGTH_BY_2];
        let mut x = [0.0f32; FFT_LENGTH_BY_2];
        for v in x_old.iter_mut() {
            *v = rng.next_f32() * 2000.0 - 1000.0;
        }
        for v in x.iter_mut() {
            *v = rng.next_f32() * 2000.0 - 1000.0;
        }
        let mut concat = [0.0f32; FFT_LENGTH];
        concat[..FFT_LENGTH_BY_2].copy_from_slice(&x_old);
        concat[FFT_LENGTH_BY_2..].copy_from_slice(&x);

        let mut X = FftData::default();
        fft.padded_fft(&x, &x_old, Window::Rectangular, &mut X);
        for k in [0usize, 1, 16, 33, 64] {
            let (r, i) = dft_reference(&concat, k);
            let scale = r.abs().max(i.abs()).max(1.0);
            assert!(
                (X.re[k] as f64 - r).abs() < 1e-3 * scale,
                "k={} re {} vs {}",
                k,
                X.re[k],
                r
            );
            assert!(
                (X.im[k] as f64 - i).abs() < 1e-3 * scale,
                "k={} im {} vs {}",
                k,
                X.im[k],
                i
            );
        }
    }

    /// ZeroPaddedFft(Hanning)：谱对称性/有限性基本检查。
    #[test]
    fn zero_padded_fft_basic() {
        let mut fft = Aec3Fft::new();
        let x = [1000.0f32; FFT_LENGTH_BY_2];
        let mut X = FftData::default();
        fft.zero_padded_fft(&x, Window::Hanning, &mut X);
        // 加窗后非零 → 谱非零且有限
        let mut ps = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        X.spectrum(&mut ps);
        assert!(ps.iter().all(|v| v.is_finite()));
        assert!(ps[16] > 0.0);
    }

    /// 窗表完整性：端点、对称性、峰值位置。
    #[test]
    fn window_tables() {
        assert_eq!(HANNING_64[0], 0.0);
        assert_eq!(HANNING_64[FFT_LENGTH_BY_2 - 1], 0.0);
        assert!((HANNING_64[31] - HANNING_64[32]).abs() < 1e-6); // 对称中心
        assert!((HANNING_64[31] - 0.99937846).abs() < 1e-6);
        assert!(SQRT_HANNING_128[0] == 0.0);
        assert_eq!(SQRT_HANNING_128[64], 1.0); // 峰值在中心
        assert!((SQRT_HANNING_128[1] - 0.02454122852291).abs() < 1e-7);
        // 对称性
        for k in 1..64 {
            assert!((SQRT_HANNING_128[k] - SQRT_HANNING_128[128 - k]).abs() < 1e-6);
        }
    }
}
