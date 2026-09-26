//! 残余回声 R2 估计器，对照
//! `modules/audio_processing/aec3/residual_echo_estimator.{h,cc}`。
//!
//! 双模式：
//! - 线性可信：`R2 = S2_linear/Erle`（有界）、`R2_unbounded = S2_linear/ErleUnbounded`
//!   + 各加线性混响尾；饱和回声 → 两者 = Y2；
//! - 非线性模式：`X2 = 3 块窗（delay±1）渲染功率逐 bin max`，软噪声门，
//!   减平稳噪声底 ×10，`R2 = echo_path_gain·X2` + 非线性混响尾；
//!   透明模式（本移植恒不激活）gain = 1e-4。
//!
//! ML 注入点：注入且初始化且线性可信 → 模型输出直接覆盖 R2/R2_unbounded。

use crate::constants::{
    BLOCK_SIZE, ECHO_MODEL_MODEL_REVERB_IN_NONLINEAR_MODE,
    ECHO_MODEL_NOISE_FLOOR_HOLD, ECHO_MODEL_NOISE_FLOOR_LEAK,
    ECHO_MODEL_NOISE_GATE_POWER, ECHO_MODEL_NOISE_GATE_SLOPE,
    ECHO_MODEL_MIN_NOISE_FLOOR_POWER, ECHO_MODEL_RENDER_POST_WINDOW_SIZE,
    ECHO_MODEL_RENDER_PRE_WINDOW_SIZE, ECHO_MODEL_STATIONARY_GATE_SLOPE,
    EP_STRENGTH_DEFAULT_GAIN, FFT_LENGTH_BY_2_PLUS_1, TRANSPARENT_MODE_GAIN,
};
use crate::delay_estimate::DelayEstimate;
use crate::neural_residual_echo_estimator::{MlReeState, NeuralResidualEchoEstimator};
use crate::render_delay_buffer::RenderBufferView;
use crate::reverb_model::{
    simplified_reverb_frequency_response, ReverbModel, REVERB_DECAY,
};

/// 估计所需的线性层/AecState 汇总输入。
pub struct EstimateContext<'a> {
    pub usable_linear_estimate: bool,
    pub saturated_echo: bool,
    pub min_direct_path_filter_delay_blocks: i32,
    /// 精滤波器当前分区数（线性混响起始 = 分区数+1）。
    pub filter_length_blocks: usize,
    /// 透明模式（本移植恒 false）。
    pub transparent_mode_active: bool,
    pub erle: &'a [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub erle_unbounded: &'a [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// ML 用：外部延迟估计（块）。
    pub external_delay_blocks: Option<DelayEstimate>,
}

/// 残余回声估计器（`ResidualEchoEstimator`）。单声道。
pub struct ResidualEchoEstimator {
    echo_reverb: ReverbModel,
    x2_noise_floor: [f32; FFT_LENGTH_BY_2_PLUS_1],
    x2_noise_floor_counter: [usize; FFT_LENGTH_BY_2_PLUS_1],
    ml_ree_state: MlReeState,
    neural: Option<Box<dyn NeuralResidualEchoEstimator>>,
    // 复用缓冲
    x2: [f32; FFT_LENGTH_BY_2_PLUS_1],
    reverb_scaling: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for ResidualEchoEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidualEchoEstimator {
    pub fn new() -> Self {
        Self::with_neural(None)
    }

    /// 注入 ML 估计器（`with_neural_residual_echo_estimator`）。
    pub fn with_neural(neural: Option<Box<dyn NeuralResidualEchoEstimator>>) -> Self {
        Self {
            echo_reverb: ReverbModel::new(),
            x2_noise_floor: [ECHO_MODEL_MIN_NOISE_FLOOR_POWER; FFT_LENGTH_BY_2_PLUS_1],
            x2_noise_floor_counter: [ECHO_MODEL_NOISE_FLOOR_HOLD; FFT_LENGTH_BY_2_PLUS_1],
            ml_ree_state: MlReeState::Uninitialized,
            neural,
            x2: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            reverb_scaling: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    pub fn ml_ree_state(&self) -> MlReeState {
        self.ml_ree_state
    }

    /// ML 是否激活（EchoRemover 据此切换输出选择与 Suppressor 配置）。
    pub fn ml_ree_is_active(&self) -> bool {
        self.ml_ree_state == MlReeState::Active
    }

    /// 每块估计（`ResidualEchoEstimator::Estimate`）。
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &mut self,
        ctx: &EstimateContext,
        render_buffer: &RenderBufferView,
        capture: &[f32; BLOCK_SIZE],
        linear_aec_output: &[f32; BLOCK_SIZE],
        s2_linear: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        e2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        filter_frequency_responses: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        dominant_nearend: bool,
        r2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        r2_unbounded: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        // ML 状态机：注入且初始化时，线性可信才激活。
        if let Some(neural) = self.neural.as_ref() {
            if neural.is_initialized() {
                self.ml_ree_state = if ctx.usable_linear_estimate {
                    MlReeState::Active
                } else {
                    MlReeState::Initialized
                };
            }
        }

        // 渲染平稳噪声底（最小统计）。
        self.update_render_noise_power(render_buffer);

        // ML：即使稍后被覆盖也持续运行（保证模型看到连续等时率信号）。
        if self.ml_ree_state != MlReeState::Uninitialized {
            const NEURAL_DELAY_HEADROOM_BLOCKS: i32 = 3; // 12ms / 4ms
            const JITTER_MARGIN_BLOCKS: i32 = 3;
            let headroom_render_buffer = render_buffer.headroom();
            let mut headroom_blocks = 0i32;
            if let Some(ext) = ctx.external_delay_blocks.as_ref() {
                if ext.delay as i32 > NEURAL_DELAY_HEADROOM_BLOCKS + JITTER_MARGIN_BLOCKS
                    && headroom_render_buffer > 0
                {
                    headroom_blocks = (headroom_render_buffer as i32 - 1)
                        .min(NEURAL_DELAY_HEADROOM_BLOCKS);
                }
            }
            let render = *render_buffer.get_block(headroom_blocks as isize);
            self.neural.as_mut().unwrap().estimate(
                &render,
                capture,
                linear_aec_output,
                s2_linear,
                y2,
                e2,
                dominant_nearend,
                r2,
                r2_unbounded,
            );
        }

        // 传统路径。
        if self.ml_ree_state != MlReeState::Active {
            if ctx.usable_linear_estimate {
                if ctx.saturated_echo {
                    // 饱和回声：假设残余与麦克风同谱。
                    r2.copy_from_slice(y2);
                    r2_unbounded.copy_from_slice(y2);
                } else {
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        r2[k] = s2_linear[k] / ctx.erle[k];
                        r2_unbounded[k] = s2_linear[k] / ctx.erle_unbounded[k];
                    }
                }
                self.update_reverb(true, ctx, render_buffer, filter_frequency_responses);
                self.echo_reverb.add_reverb(r2);
                self.echo_reverb.add_reverb(r2_unbounded);
            } else {
                let echo_path_gain = Self::echo_path_gain(ctx.transparent_mode_active);
                if ctx.saturated_echo {
                    r2.copy_from_slice(y2);
                    r2_unbounded.copy_from_slice(y2);
                } else {
                    // 回声生成功率：delay ± 1 块窗口的逐 bin max。
                    self.echo_generating_power(
                        render_buffer,
                        ctx.min_direct_path_filter_delay_blocks,
                    );
                    self.apply_noise_gate();
                    // 减平稳噪声底，避免平稳渲染噪声引起过度抑制。
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        self.x2[k] -= ECHO_MODEL_STATIONARY_GATE_SLOPE * self.x2_noise_floor[k];
                        self.x2[k] = self.x2[k].max(0.0);
                    }
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        r2[k] = self.x2[k] * echo_path_gain;
                        r2_unbounded[k] = self.x2[k] * echo_path_gain;
                    }
                }
                if ECHO_MODEL_MODEL_REVERB_IN_NONLINEAR_MODE
                    && !ctx.transparent_mode_active
                {
                    self.update_reverb(false, ctx, render_buffer, filter_frequency_responses);
                    self.echo_reverb.add_reverb(r2);
                    self.echo_reverb.add_reverb(r2_unbounded);
                }
            }
        }
        // 未移植：stationarity 的 residual_scaling（默认关）。
    }

    /// 回声路径增益（功率域，`GetEchoPathGain`）。
    fn echo_path_gain(transparent: bool) -> f32 {
        // early/late 反射在本默认配置下同值。
        let amplitude = if transparent {
            TRANSPARENT_MODE_GAIN
        } else {
            EP_STRENGTH_DEFAULT_GAIN
        };
        amplitude * amplitude
    }

    /// 3 块窗回声生成功率（`EchoGeneratingPower`，单声道）。
    fn echo_generating_power(
        &mut self,
        render_buffer: &RenderBufferView,
        filter_delay_blocks: i32,
    ) {
        let window_start = (filter_delay_blocks as isize
            - ECHO_MODEL_RENDER_PRE_WINDOW_SIZE as isize)
            .max(0);
        let window_end =
            filter_delay_blocks + ECHO_MODEL_RENDER_POST_WINDOW_SIZE as i32;
        self.x2.fill(0.0);
        let spectra = render_buffer.spectrum_buffer_ref();
        let mut idx = spectra
            .offset_index(spectra.read(), window_start as isize);
        let stop = spectra.offset_index(spectra.read(), window_end as isize + 1);
        while idx != stop {
            let s = spectra.get(idx);
            for j in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.x2[j] = self.x2[j].max(s[j]);
            }
            idx = spectra.inc_index(idx);
        }
    }

    /// 软噪声门（`ApplyNoiseGate`）。
    fn apply_noise_gate(&mut self) {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if ECHO_MODEL_NOISE_GATE_POWER > self.x2[k] {
                self.x2[k] = (self.x2[k]
                    - ECHO_MODEL_NOISE_GATE_SLOPE
                        * (ECHO_MODEL_NOISE_GATE_POWER - self.x2[k]))
                .max(0.0);
            }
        }
    }

    /// 渲染平稳噪声底（`UpdateRenderNoisePower`，最小统计）。
    fn update_render_noise_power(&mut self, render_buffer: &RenderBufferView) {
        let x2 = render_buffer.spectrum(0);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if x2[k] < self.x2_noise_floor[k] {
                self.x2_noise_floor[k] = x2[k];
                self.x2_noise_floor_counter[k] = 0;
            } else if self.x2_noise_floor_counter[k] >= ECHO_MODEL_NOISE_FLOOR_HOLD {
                self.x2_noise_floor[k] = (self.x2_noise_floor[k] * ECHO_MODEL_NOISE_FLOOR_LEAK)
                    .max(ECHO_MODEL_MIN_NOISE_FLOOR_POWER);
            } else {
                self.x2_noise_floor_counter[k] += 1;
            }
        }
    }

    /// 混响更新（`UpdateReverb`）。
    fn update_reverb(
        &mut self,
        linear: bool,
        ctx: &EstimateContext,
        render_buffer: &RenderBufferView,
        filter_frequency_responses: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        let first_partition = if linear {
            ctx.filter_length_blocks + 1
        } else {
            ctx.min_direct_path_filter_delay_blocks.max(0) as usize + 1
        };
        let power = render_buffer.spectrum(first_partition as isize);
        if linear {
            self.reverb_scaling = simplified_reverb_frequency_response(
                filter_frequency_responses,
                ctx.min_direct_path_filter_delay_blocks.max(0) as usize,
            );
            self.echo_reverb
                .update_reverb(power, &self.reverb_scaling, REVERB_DECAY);
        } else {
            let gain = Self::echo_path_gain(ctx.transparent_mode_active);
            self.echo_reverb
                .update_reverb_no_freq_shaping(power, gain, REVERB_DECAY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_delay_buffer::RenderDelayBuffer;

    fn setup_render(n_blocks: usize, amp: f32) -> RenderDelayBuffer {
        let mut buf = RenderDelayBuffer::new();
        let mut state = 0xABCDu64;
        for _ in 0..n_blocks {
            let mut b = [0.0f32; BLOCK_SIZE];
            for v in b.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *v = ((state >> 33) as f32 / (1u64 << 30) as f32 - 1.0) * amp;
            }
            buf.insert(&b);
            buf.prepare_capture_processing();
        }
        buf.align_from_delay(5);
        buf
    }

    fn ctx(usable: bool, saturated: bool) -> EstimateContext<'static> {
        EstimateContext {
            usable_linear_estimate: usable,
            saturated_echo: saturated,
            min_direct_path_filter_delay_blocks: 2,
            filter_length_blocks: 12,
            transparent_mode_active: false,
            erle: &[1.0; 65],
            erle_unbounded: &[1.0; 65],
            external_delay_blocks: None,
        }
    }

    /// 线性模式：R2 = S2/ERLE；饱和 → R2 = Y2。
    #[test]
    fn linear_and_saturated_modes() {
        let buf = setup_render(60, 3000.0);
        let mut est = ResidualEchoEstimator::new();
        let cap = [0.0f32; BLOCK_SIZE];
        let s2 = [1.0e6f32; 65];
        let y2 = [2.0e6f32; 65];
        let e2 = [1.0e5f32; 65];
        let h2 = vec![[0.1f32; 65]; 13];
        let mut r2 = [0.0f32; 65];
        let mut r2u = [0.0f32; 65];
        est.estimate(
            &ctx(true, false),
            &buf.get_render_buffer(),
            &cap,
            &cap,
            &s2,
            &y2,
            &e2,
            &h2,
            false,
            &mut r2,
            &mut r2u,
        );
        for k in [1usize, 16, 64] {
            // ERLE=1 时 R2 = S2 + 混响尾（WGN 渲染有可观的 reverb 贡献）
            assert!(
                r2[k] >= 1.0e6 && r2[k] < 5.0e7,
                "k={} r2={} (应 ≥S2 且有限)",
                k,
                r2[k]
            );
            assert!(r2[k] > 0.0 && r2[k].is_finite());
        }
        // 饱和 → R2 = Y2
        let mut r2b = [0.0f32; 65];
        let mut r2ub = [0.0f32; 65];
        est.estimate(
            &ctx(true, true),
            &buf.get_render_buffer(),
            &cap,
            &cap,
            &s2,
            &y2,
            &e2,
            &h2,
            false,
            &mut r2b,
            &mut r2ub,
        );
        // 饱和 → R2 = Y2 + 混响尾（AddReverb 照常执行，与源码一致）
        assert!(r2b[16] >= 2.0e6);
        assert!((r2b[16] - r2ub[16]).abs() < 1.0);
    }

    /// 非线性模式：R2 = gain·门控 X2（增益 1.0，强渲染 → R2 > 0）。
    #[test]
    fn nonlinear_mode() {
        let buf = setup_render(60, 3000.0);
        let mut est = ResidualEchoEstimator::new();
        let cap = [0.0f32; BLOCK_SIZE];
        let zeros = [0.0f32; 65];
        let h2 = vec![[0.0f32; 65]; 13];
        let mut r2 = [0.0f32; 65];
        let mut r2u = [0.0f32; 65];
        est.estimate(
            &ctx(false, false),
            &buf.get_render_buffer(),
            &cap,
            &cap,
            &zeros,
            &zeros,
            &zeros,
            &h2,
            false,
            &mut r2,
            &mut r2u,
        );
        // 强 WGN 渲染：过噪声门/floor 后应剩可观的回声估计
        assert!(r2[16] > 1.0e6, "r2[16]={}", r2[16]);
        assert!((r2[16] - r2u[16]).abs() < 1e-3, "非线性模式两者相同");
    }

    /// X2 噪声底：静默渲染下 floor 上漂被钳位在初值下方。
    #[test]
    fn render_noise_floor_falls_on_silence() {
        let mut buf = RenderDelayBuffer::new();
        for _ in 0..60 {
            buf.insert(&[0.0; BLOCK_SIZE]);
            buf.prepare_capture_processing();
        }
        buf.align_from_delay(5);
        let mut est = ResidualEchoEstimator::new();
        let cap = [0.0f32; BLOCK_SIZE];
        let zeros = [0.0f32; 65];
        let h2 = vec![[0.0f32; 65]; 13];
        let mut r2 = [0.0f32; 65];
        let mut r2u = [0.0f32; 65];
        est.estimate(
            &ctx(false, false),
            &buf.get_render_buffer(),
            &cap,
            &cap,
            &zeros,
            &zeros,
            &zeros,
            &h2,
            false,
            &mut r2,
            &mut r2u,
        );
        // 静默 + floor 全额扣除 → R2 = 0
        assert_eq!(r2[16], 0.0);
    }
}
