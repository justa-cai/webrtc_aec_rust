//! 线性回声消除器（`EchoRemover` 的线性部分），对照
//! `modules/audio_processing/aec3/echo_remover.{h,cc}`。
//!
//! 只保留线性路径：Subtractor 编排、refined/coarse 输出选择与 30 样本交叉淡化、
//! Y2/E2/S2_linear 谱计算、linear_output 导出、路径变化复位分发。
//!
//! 未移植（非线性部分）：SuppressionGain、ComfortNoiseGenerator、
//! SuppressionFilter、ResidualEchoEstimator、ERLE/ERLE 指标。

use crate::aec_state::AecState;
use crate::comfort_noise_generator::ComfortNoiseGenerator;
use crate::constants::SuppressorConfig;
use crate::erle_estimator::ErleEstimator;
use crate::residual_echo_estimator::{EstimateContext, ResidualEchoEstimator};
use crate::suppression_filter::SuppressionFilter;
use crate::suppression_gain::SuppressionGain;
use crate::constants::{
    BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1, SIGNAL_TRANSITION_SIZE, USE_REFINED_COARSE_RATIO,
    USE_REFINED_S2_THRESHOLD, USE_REFINED_Y2_THRESHOLD,
};
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::fft::Aec3Fft;
use crate::fft_data::FftData;
use crate::render_delay_buffer::RenderBufferView;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::subtractor::Subtractor;
use crate::subtractor_output::SubtractorOutput;

/// `UseRefinedOutput`（echo_remover.cc:112）。
fn use_refined_output(o: &SubtractorOutput) -> bool {
    // coarse 明显更优（0.9 系数留裕度）且信号能量足够时选 coarse。
    if o.e2_coarse < USE_REFINED_COARSE_RATIO * o.e2_refined
        && o.y2 > USE_REFINED_Y2_THRESHOLD
        && (o.s2_refined > USE_REFINED_S2_THRESHOLD || o.s2_coarse > USE_REFINED_S2_THRESHOLD)
    {
        return false;
    }
    // refined 发散（误差能量超过麦克风能量）时选功率更小的一级。
    if o.e2_coarse < o.e2_refined && o.y2 < o.e2_refined {
        return false;
    }
    true
}

/// 30 样本交叉淡化（`SignalTransition`，echo_remover.cc:76）。
fn signal_transition(from: &[f32; BLOCK_SIZE], to: &[f32; BLOCK_SIZE], out: &mut [f32; BLOCK_SIZE]) {
    if std::ptr::eq(from.as_ptr(), to.as_ptr()) {
        out.copy_from_slice(to);
    } else {
        let one_by = 1.0 / (SIGNAL_TRANSITION_SIZE as f32 + 1.0);
        for k in 0..SIGNAL_TRANSITION_SIZE {
            let a = (k + 1) as f32 * one_by;
            out[k] = a * to[k] + (1.0 - a) * from[k];
        }
        out[SIGNAL_TRANSITION_SIZE..].copy_from_slice(&to[SIGNAL_TRANSITION_SIZE..]);
    }
}

/// 每块处理的输出谱与诊断（线性部分）。
#[derive(Clone, Debug)]
pub struct BlockResult {
    /// 线性滤波器选择的输出（交叉淡化后）。
    pub e: [f32; BLOCK_SIZE],
    /// |Y[k]|²（麦克风 sqrt-Hann 谱）。
    pub y2: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// |E[k]|²。
    pub e2: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// |Y−E|²：线性回声功率谱估计（S2_linear）。
    pub s2_linear: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// 本块是否选择了 refined 输出。
    pub use_refined: bool,
    /// 延迟变化事件。
    pub delay_change: bool,
    /// 本块近端主导检测状态（GetGain 之后）。
    pub nearend_state: bool,
}

impl Default for BlockResult {
    fn default() -> Self {
        Self {
            e: [0.0; BLOCK_SIZE],
            y2: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e2: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            s2_linear: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            use_refined: true,
            delay_change: false,
            nearend_state: false,
        }
    }
}

/// 线性 EchoRemover（`EchoRemoverImpl` 线性部分）。单声道。
pub struct EchoRemover {
    subtractor: Subtractor,
    aec_state: AecState,
    render_signal_analyzer: RenderSignalAnalyzer,
    fft: Aec3Fft,
    y_old: [f32; BLOCK_SIZE],
    e_old: [f32; BLOCK_SIZE],
    refined_filter_output_last_selected: bool,
    block_counter: usize,
    // 复用缓冲
    y_fft: FftData,
    e_fft: FftData,
    output: SubtractorOutput,
    // —— 非线性层 ——
    suppressor_config: SuppressorConfig,
    cng: ComfortNoiseGenerator,
    residual_echo_estimator: ResidualEchoEstimator,
    suppression_gain: SuppressionGain,
    suppression_filter: SuppressionFilter,
    erle_estimator: ErleEstimator,
    /// ML 激活时使用的替换 Suppressor 配置（构造时由 adjust_config 生成）。
    ml_ree_suppressor_config: Option<SuppressorConfig>,
    ml_ree_was_active: bool,
    // 复用缓冲
    r2: [f32; FFT_LENGTH_BY_2_PLUS_1],
    r2_unbounded: [f32; FFT_LENGTH_BY_2_PLUS_1],
    comfort_noise: FftData,
    high_band_comfort_noise: FftData,
    g: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for EchoRemover {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoRemover {
    pub fn new() -> Self {
        Self::with_suppressor_config(SuppressorConfig::default())
    }

    /// 注入抑制器配置（双讲调优实验用）。
    pub fn with_suppressor_config(suppressor_config: SuppressorConfig) -> Self {
        Self {
            suppressor_config,
            subtractor: Subtractor::new(),
            aec_state: AecState::new(),
            render_signal_analyzer: RenderSignalAnalyzer::new(),
            fft: Aec3Fft::new(),
            y_old: [0.0; BLOCK_SIZE],
            e_old: [0.0; BLOCK_SIZE],
            refined_filter_output_last_selected: true,
            block_counter: 0,
            y_fft: FftData::default(),
            e_fft: FftData::default(),
            output: SubtractorOutput::new(),
            cng: ComfortNoiseGenerator::new(),
            residual_echo_estimator: ResidualEchoEstimator::new(),
            suppression_gain: SuppressionGain::with_config(&suppressor_config),
            suppression_filter: SuppressionFilter::new(),
            erle_estimator: ErleEstimator::new(),
            ml_ree_suppressor_config: None,
            ml_ree_was_active: false,
            r2: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            r2_unbounded: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            comfort_noise: FftData::default(),
            high_band_comfort_noise: FftData::default(),
            g: [1.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    pub fn aec_state(&self) -> &AecState {
        &self.aec_state
    }

    pub fn subtractor(&self) -> &Subtractor {
        &self.subtractor
    }

    /// 处理一块（`EchoRemoverImpl::ProcessCapture` 线性部分）。
    ///
    /// `capture` 原地替换为线性 AEC 输出；`linear_output` 可选导出 e（与输出相同）。
    #[allow(clippy::too_many_arguments)]
    pub fn process_capture(
        &mut self,
        echo_path_variability: &EchoPathVariability,
        capture_signal_saturation: bool,
        external_delay: Option<crate::delay_estimate::DelayEstimate>,
        render_buffer: &RenderBufferView,
        linear_output: Option<&mut [f32; BLOCK_SIZE]>,
        capture: &mut [f32; BLOCK_SIZE],
    ) -> BlockResult {
        self.block_counter += 1;
        let y_snapshot = *capture;

        self.aec_state.update_capture_saturation(capture_signal_saturation);

        let mut result = BlockResult::default();

        if echo_path_variability.audio_path_changed() {
            self.subtractor.handle_echo_path_change(echo_path_variability);
            self.aec_state.handle_echo_path_change(echo_path_variability);
            result.delay_change =
                echo_path_variability.delay_change != DelayAdjustment::None;
            if result.delay_change {
                self.suppression_gain.set_initial_state(true);
            }
        }

        // render 信号分析
        self.render_signal_analyzer.update(
            render_buffer,
            Some(self.aec_state.min_direct_path_filter_delay().max(0) as usize),
        );

        // 初始状态切换：滤波器加长、增益配置转稳态
        if self.aec_state.transition_triggered() {
            self.subtractor.exit_initial_state();
            self.suppression_gain.set_initial_state(false);
        }

        // 线性回声消除
        self.subtractor.process(
            render_buffer,
            &y_snapshot,
            &self.render_signal_analyzer,
            self.aec_state.saturated_capture(),
            &mut self.output,
        );

        // ML 激活时禁用 coarse 输出选择（echo_remover.cc:430-441）
        let ml_ree_is_active = self.residual_echo_estimator.ml_ree_is_active();
        let use_refined = if ml_ree_is_active {
            true
        } else {
            use_refined_output(&self.output)
        };
        {
            let from = if self.refined_filter_output_last_selected {
                &self.output.e_refined
            } else {
                &self.output.e_coarse
            };
            let to = if use_refined {
                &self.output.e_refined
            } else {
                &self.output.e_coarse
            };
            signal_transition(from, to, &mut result.e);
        }
        self.refined_filter_output_last_selected = use_refined;
        result.use_refined = use_refined;

        // Y/E 谱（sqrt-Hann 加窗 PaddedFft）与 S2_linear
        let e_snapshot = result.e;
        self.fft
            .windowed_padded_fft(&y_snapshot, &mut self.y_old, &mut self.y_fft);
        self.fft
            .windowed_padded_fft(&e_snapshot, &mut self.e_old, &mut self.e_fft);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            result.s2_linear[k] = (self.y_fft.re[k] - self.e_fft.re[k])
                * (self.y_fft.re[k] - self.e_fft.re[k])
                + (self.y_fft.im[k] - self.e_fft.im[k])
                    * (self.y_fft.im[k] - self.e_fft.im[k]);
        }
        self.y_fft.spectrum(&mut result.y2);
        self.e_fft.spectrum(&mut result.e2);

        // 输出与线性导出
        capture.copy_from_slice(&e_snapshot);
        if let Some(lo) = linear_output {
            lo.copy_from_slice(&e_snapshot);
        }

        // AEC 状态更新（在 E2 clamp 之前，与源码一致）
        let freq_responses = self.subtractor.filter_frequency_responses().clone();
        let impulse = self.subtractor.filter_impulse_responses().clone();
        self.aec_state.update(
            external_delay,
            &impulse,
            render_buffer,
            &self.output.e2_refined_spectrum,
            &result.y2,
            &self.output,
        );

        // ——————— 非线性层（echo_remover.cc:482-530） ———————
        let usable = self.aec_state.usable_linear_estimate();
        // ERLE 更新（clamp 前，双讲冻结；上一块的近端检测态）
        let dominant_nearend_prev = self.suppression_gain.is_dominant_nearend();
        self.erle_estimator.update(
            self.aec_state.converged_filters(),
            dominant_nearend_prev,
            &result.y2,
            &self.output.e2_refined_spectrum,
        );

        // 舒适噪声（:482）—— nearend 谱用未 clamp 的 E2/Y2。
        let nearend_spectrum_preclamp: [f32; FFT_LENGTH_BY_2_PLUS_1] = if usable {
            result.e2
        } else {
            result.y2
        };
        self.cng.compute(
            self.aec_state.saturated_capture(),
            &nearend_spectrum_preclamp,
            &mut self.comfort_noise,
            &mut self.high_band_comfort_noise,
        );

        // 残余回声估计（:490）。y_old/e_old 此时已是当前块（与源码一致）。
        let y_current = self.y_old;
        let e_current = self.e_old;
        let ctx = EstimateContext {
            usable_linear_estimate: usable,
            saturated_echo: self.aec_state.saturated_echo(),
            min_direct_path_filter_delay_blocks: self
                .aec_state
                .min_direct_path_filter_delay(),
            filter_length_blocks: self.subtractor.size_partitions(),
            transparent_mode_active: false, // 透明模式恒不激活（README 偏差清单）
            erle: self.erle_estimator.erle(),
            erle_unbounded: self.erle_estimator.erle_unbounded(),
            external_delay_blocks: external_delay,
        };
        let s2_linear_copy = result.s2_linear;
        let y2_copy = result.y2;
        let mut e2_copy = result.e2;
        self.residual_echo_estimator.estimate(
            &ctx,
            render_buffer,
            &y_current,
            &e_current,
            &s2_linear_copy,
            &y2_copy,
            &e2_copy,
            &freq_responses,
            dominant_nearend_prev,
            &mut self.r2,
            &mut self.r2_unbounded,
        );

        // E2 就地 clamp 到 Y2 以下（:495，仅 UsableLinearEstimate）。
        if usable {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                e2_copy[k] = e2_copy[k].min(y2_copy[k]);
            }
        }
        let nearend_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1] = if usable {
            e2_copy
        } else {
            y2_copy
        };

        // 抑制增益（:524）。ML 激活时使用替换配置。
        let active_suppressor_config: SuppressorConfig = if ml_ree_is_active {
            self.ml_ree_suppressor_config.unwrap_or(self.suppressor_config)
        } else {
            self.suppressor_config
        };
        // ML 激活沿会触发上游 UpdateStateDependingOnConfig；本实现无 ML 注入
        // 时恒为 false，配置不变。
        let config_changed = ml_ree_is_active != self.ml_ree_was_active;
        let _ = config_changed;
        self.ml_ree_was_active = ml_ree_is_active;
        let render_block = *render_buffer.get_block(0);
        let _high_bands_gain = self.suppression_gain.get_gain(
            &active_suppressor_config,
            &nearend_spectrum,
            &self.r2,
            &self.r2_unbounded,
            self.cng.noise_spectrum(),
            self.aec_state.saturated_echo(),
            &render_block,
            echo_path_variability.clock_drift,
            &mut self.g,
        );
        result.nearend_state = self.suppression_gain.is_dominant_nearend();
        // 16 kHz 单带：高带增益恒 1.0（多带支持见 README 偏差清单）。

        // 增益应用（:529）：UseLinearFilterOutput ? E : Y 作为低带输入。
        // capture 此时为线性输出 e —— 原地覆写为最终 NLP 输出。
        let spectrum_in = if usable {
            self.e_fft
        } else {
            self.y_fft
        };
        let mut e_out = *capture;
        self.suppression_filter
            .apply_gain(&self.comfort_noise, &self.g, &spectrum_in, &mut e_out);
        capture.copy_from_slice(&e_out);

        result
    }
}
