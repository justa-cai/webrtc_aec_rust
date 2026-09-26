//! 简化噪声抑制（后挂验证模块），对照
//! `modules/audio_processing/ns/noise_suppressor.cc` 的骨架。
//!
//! **验证用途**：确认"NS 后挂能吃掉多少 AEC 残留"，成功后再做完整移植
//! （完整版含三特征语音概率/分位数噪声估计/能量缩放，见 README 偏差清单）。
//!
//! 结构与 webrtc NS 相同：10ms 帧（160）/ FFT 256 / overlap 96 /
//! 混合窗（96 上升 Hann + 64 平坦 + 96 下降，单边加窗 + 直接 OLA，COLA=1）。
//! 简化点：噪声估计用最小统计（快下慢上+语音冻结）；语音概率硬编码
//! （Y > 1.5·N 判语音）；Wiener `H = ξ/(1+ξ)` 决策导向 α=0.98，floor=档位。

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// NS 档位（对应 webrtc NS level）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NsLevel {
    /// 6 dB（floor 0.5）
    Low,
    /// 12 dB（floor 0.25）——webrtc 默认
    Moderate,
    /// 18 dB（floor 0.125）
    High,
}

impl NsLevel {
    fn floor(self) -> f32 {
        match self {
            NsLevel::Low => 0.5,
            NsLevel::Moderate => 0.25,
            NsLevel::High => 0.125,
        }
    }
}

const NS_FRAME: usize = 160;
const NS_FFT: usize = 256;
const NS_OVERLAP: usize = 96;
const NS_BINS: usize = 129;

/// 混合窗前半（96 值，照抄 noise_suppressor.cc:56-88 的
/// kBlocks160w256FirstHalf；窗形 = 上升 Hann + 平坦 + 对称下降）。
const W96: [f32; 96] = [
    0.00000000, 0.01636173, 0.03271908, 0.04906767, 0.06540313, 0.08172107,
    0.09801714, 0.11428696, 0.13052619, 0.14673047, 0.16289547, 0.17901686,
    0.19509032, 0.21111155, 0.22707626, 0.24298018, 0.25881905, 0.27458862,
    0.29028468, 0.30590302, 0.32143947, 0.33688985, 0.35225005, 0.36751594,
    0.38268343, 0.39774847, 0.41270703, 0.42755509, 0.44228869, 0.45690388,
    0.47139674, 0.48576339, 0.50000000, 0.51410274, 0.52806785, 0.54189158,
    0.55557023, 0.56910015, 0.58247770, 0.59569930, 0.60876143, 0.62166057,
    0.63439328, 0.64695615, 0.65934582, 0.67155895, 0.68359230, 0.69544264,
    0.70710678, 0.71858162, 0.72986407, 0.74095113, 0.75183981, 0.76252720,
    0.77301045, 0.78328675, 0.79335334, 0.80320753, 0.81284668, 0.82226822,
    0.83146961, 0.84044840, 0.84920218, 0.85772861, 0.86602540, 0.87409034,
    0.88192126, 0.88951608, 0.89687274, 0.90398929, 0.91086382, 0.91749450,
    0.92387953, 0.93001722, 0.93590593, 0.94154407, 0.94693013, 0.95206268,
    0.95694034, 0.96156180, 0.96592583, 0.97003125, 0.97387698, 0.97746197,
    0.98078528, 0.98384601, 0.98664333, 0.98917651, 0.99144486, 0.99344778,
    0.99518473, 0.99665524, 0.99785892, 0.99879546, 0.99946459, 0.99986614,
];

/// 应用混合窗（`ApplyFilterBankWindow`：0..96 上升，96..161 平坦，161..256 对称下降）。
fn apply_window(x: &mut [f32; NS_FFT]) {
    for i in 0..NS_OVERLAP {
        x[i] *= W96[i];
    }
    // 平坦段 96..161 无操作；下降段 161..256 用对称窗。
    for i in 161..NS_FFT {
        x[i] *= W96[NS_FFT - i];
    }
}

/// 简化噪声抑制器。单声道、16 kHz。
pub struct SimpleNs {
    forward: std::sync::Arc<dyn Fft<f32>>,
    inverse: std::sync::Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    buf: Vec<Complex<f32>>,
    /// 分析侧旧数据（96 样本）。
    old_data: [f32; NS_OVERLAP],
    /// 合成侧 overlap 记忆。
    synthesis_memory: [f32; NS_OVERLAP],
    /// 噪声谱估计（幅度域）。
    noise: [f32; NS_BINS],
    /// 时域平滑谱（降低逐 bin 方差，稳定噪声跟踪）。
    y_smooth: [f32; NS_BINS],
    /// 上一帧谱与增益（决策导向）。
    prev_spectrum: [f32; NS_BINS],
    prev_gain: [f32; NS_BINS],
    level: NsLevel,
    frame_counter: usize,
}

impl SimpleNs {
    pub fn new(level: NsLevel) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(NS_FFT);
        let inverse = planner.plan_fft_inverse(NS_FFT);
        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        Self {
            forward,
            inverse,
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            buf: vec![Complex::new(0.0, 0.0); NS_FFT],
            old_data: [0.0; NS_OVERLAP],
            synthesis_memory: [0.0; NS_OVERLAP],
            noise: [1e3f32; NS_BINS], // 启动保守值（webrtc 用白噪/粉噪混合）
            y_smooth: [1e3f32; NS_BINS],
            prev_spectrum: [1e3f32; NS_BINS],
            prev_gain: [1.0; NS_BINS],
            level,
            frame_counter: 0,
        }
    }

    /// 调试：打印内部状态。
    pub fn debug_print(&self) {
        for i in [1usize, 10, 40, 80, 120] {
            println!("bin {:>3}: y={:.0} n={:.0} gain={:.3}", i, self.prev_spectrum[i], self.noise[i], self.prev_gain[i]);
        }
    }

    /// 处理一帧（160 样本，原地替换）。
    pub fn process(&mut self, frame: &mut [f32; NS_FRAME]) {
        // 扩展帧 [old_data, frame] = 256，加窗，FFT。
        let mut extended = [0.0f32; NS_FFT];
        extended[..NS_OVERLAP].copy_from_slice(&self.old_data);
        extended[NS_OVERLAP..].copy_from_slice(frame);
        self.old_data.copy_from_slice(&frame[NS_FRAME - NS_OVERLAP..]);
        apply_window(&mut extended);
        for (i, v) in extended.iter().enumerate() {
            self.buf[i] = Complex::new(*v, 0.0);
        }
        self.forward.process_with_scratch(&mut self.buf, &mut self.scratch);

        // 幅度谱（+1 下限，webrtc 同式）。
        let mut y = [0.0f32; NS_BINS];
        y[0] = self.buf[0].re.abs() + 1.0;
        y[NS_BINS - 1] = self.buf[NS_FFT / 2].re.abs() + 1.0;
        for i in 1..(NS_BINS - 1) {
            y[i] = (self.buf[i].re * self.buf[i].re + self.buf[i].im * self.buf[i].im).sqrt() + 1.0;
        }

        // —— 谱平滑（降低逐 bin 方差；|FFT|² 是方差很大的指数型分布）——
        for i in 0..NS_BINS {
            self.y_smooth[i] = 0.7 * self.y_smooth[i] + 0.3 * y[i];
        }

        // —— 噪声估计：语音概率加权均值跟踪（webrtc PostUpdate 本质）——
        // 更新速度 (1−γ) = 0.1·(1−p_i)，p_i = clamp((ys/n−1)/2, 0, 1)：
        // 噪声 bin（y/n≈1）全速跟踪；语音 bin（y/n≥3）冻结。
        // 启动期（前 50 帧）无冻结全速收敛，避免初值自锁。
        self.frame_counter += 1;
        let startup = self.frame_counter < 50;
        for i in 0..NS_BINS {
            let ys = self.y_smooth[i];
            let alpha = if startup {
                0.3
            } else {
                let p = ((ys / self.noise[i] - 1.0) / 2.0).clamp(0.0, 1.0);
                0.1 * (1.0 - p)
            };
            self.noise[i] = (1.0 - alpha) * self.noise[i] + alpha * ys;
        }

        let mut gain = [1.0f32; NS_BINS];
        for i in 0..NS_BINS {
            let n = self.noise[i];

            // —— 决策导向先验 SNR + Wiener（webrtc: prevEst = Y_prev/N·H_prev − 1）——
            let snr_post = (y[i] / n - 1.0).max(0.0);
            let prev_est = (self.prev_spectrum[i] / n - 1.0).max(0.0) * self.prev_gain[i];
            let snr_prio = 0.98 * prev_est + 0.02 * snr_post;
            gain[i] = (snr_prio / (1.0 + snr_prio)).max(self.level.floor()).min(1.0);
        }
        self.prev_spectrum = y;
        self.prev_gain = gain;

        // 应用增益到复数谱（对称）。
        for i in 0..NS_BINS {
            let g = gain[i];
            self.buf[i] = Complex::new(self.buf[i].re * g, self.buf[i].im * g);
            if i > 0 && i < NS_FFT / 2 {
                let j = NS_FFT - i;
                self.buf[j] = Complex::new(self.buf[j].re * g, self.buf[j].im * g);
            }
        }

        // IFFT（rustfft 逆变换不含归一化，手动除 N）→ overlap-add。
        self.inverse.process_with_scratch(&mut self.buf, &mut self.scratch);
        let ext: [f32; NS_FFT] = {
            let mut e = [0.0f32; NS_FFT];
            for (i, v) in e.iter_mut().enumerate() {
                *v = self.buf[i].re / NS_FFT as f32;
            }
            e
        };
        for i in 0..NS_OVERLAP {
            frame[i] = self.synthesis_memory[i] + ext[i];
        }
        frame[NS_OVERLAP..].copy_from_slice(&ext[NS_OVERLAP..NS_FRAME]);
        self.synthesis_memory.copy_from_slice(&ext[NS_FRAME..]);
        for v in frame.iter_mut() {
            *v = v.clamp(-32768.0, 32767.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 纯白噪声输入：输出应被显著衰减（Wiener 压噪声）。
    #[test]
    fn attenuates_stationary_noise() {
        let mut ns = SimpleNs::new(NsLevel::Moderate);
        let mut state = 0x1234u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) as f32 / (1u64 << 30) as f32 - 1.0) * 1000.0
        };
        let (mut e_in, mut e_out) = (0.0f64, 0.0f64);
        for f in 0..1000 {
            let mut frame = [0.0f32; NS_FRAME];
            for v in frame.iter_mut() {
                *v = next();
            }
            e_in += frame.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
            ns.process(&mut frame);
            if f >= 100 {
                e_out += frame.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
            }
        }
        // 后 900 帧的输出能量（输入能量按比例）应显著低于输入
        let ratio = 10.0 * (e_out / (e_in * 0.9)).log10();
        assert!(ratio < -3.0, "噪声衰减不足: {:.1} dB", ratio);
    }

    /// 间歇语音（on/off 各 8 帧，模拟真实停顿）应基本保留。
    /// 注：持续不停的正弦会把噪声估计拖到语音电平（无停顿拉回），
    /// 这是简化语音概率的已知局限，完整版（三特征 LRT）可解。
    #[test]
    fn preserves_intermittent_speech() {
        let mut ns = SimpleNs::new(NsLevel::Moderate);
        let mut e_out = 0.0f64;
        let mut e_in_on = 0.0f64;
        let mut n_on = 0.0f64;
        for f in 0..1000 {
            let on = (f % 16) < 8;
            let mut frame = [0.0f32; NS_FRAME];
            if on {
                for (i, v) in frame.iter_mut().enumerate() {
                    *v = (2.0 * std::f32::consts::PI * 440.0
                        * (f * NS_FRAME + i) as f32
                        / 16000.0)
                        .sin()
                        * 5000.0;
                }
            }
            let e_in: f64 = frame.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
            ns.process(&mut frame);
            if f >= 100 && on {
                e_in_on += e_in;
                e_out += frame.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                n_on += 1.0;
            }
        }
        let _ = n_on;
        let ratio = 10.0 * (e_out / e_in_on).log10();
        assert!(ratio > -8.0, "间歇语音损伤过大: {:.1} dB", ratio);
    }
}
