//! 精滤波器更新增益（变步长 + 模型误差递推 + 频域泄漏），
//! 对照 `modules/audio_processing/aec3/refined_filter_update_gain.{h,cc}`。
//!
//! 核心公式（每 bin）：
//! - `mu = H_err / (0.5·H_err·X2 + P·E2_refined)`（P 为分区数；分母含近端项，
//!   天然抑制双讲；X2 < noise_gate 时 mu=0）
//! - `H_err −= 0.5·mu·X2·H_err`（激励越多模型误差估计越小）
//! - `G = mu·E_refined`
//! - `H_err += leakage·erl`（收敛/发散两级泄漏，erl=Σ|H|²），钳位 [floor, ceil]

use crate::constants::{
    CONFIG_CHANGE_DURATION_BLOCKS, FFT_LENGTH_BY_2_PLUS_1, REFINED_ERROR_CEIL,
    REFINED_ERROR_FLOOR, REFINED_INITIAL_LEAKAGE_CONVERGED, REFINED_INITIAL_LEAKAGE_DIVERGED,
    REFINED_INITIAL_NOISE_GATE, REFINED_LEAKAGE_CONVERGED, REFINED_LEAKAGE_DIVERGED,
    REFINED_NOISE_GATE,
};
use crate::echo_path_variability::EchoPathVariability;
use crate::fft_data::FftData;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::subtractor_output::SubtractorOutput;

/// `H_error` 初值（`kHErrorInitial`）。
const H_ERROR_INITIAL: f32 = 10000.0;
/// `poor_excitation_counter_` 初值（`kPoorExcitationCounterInitial`）。
const POOR_EXCITATION_COUNTER_INITIAL: usize = 1000;

/// 精滤波器增益配置（`EchoCanceller3Config::Filter::RefinedConfiguration`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinedConfig {
    pub leakage_converged: f32,
    pub leakage_diverged: f32,
    pub error_floor: f32,
    pub error_ceil: f32,
    pub noise_gate: f32,
}

impl RefinedConfig {
    /// 稳态配置（`filter.refined`）。
    pub fn normal() -> Self {
        Self {
            leakage_converged: REFINED_LEAKAGE_CONVERGED,
            leakage_diverged: REFINED_LEAKAGE_DIVERGED,
            error_floor: REFINED_ERROR_FLOOR,
            error_ceil: REFINED_ERROR_CEIL,
            noise_gate: REFINED_NOISE_GATE,
        }
    }
    /// 初始（未收敛期）配置（`filter.refined_initial`）。
    pub fn initial() -> Self {
        Self {
            leakage_converged: REFINED_INITIAL_LEAKAGE_CONVERGED,
            leakage_diverged: REFINED_INITIAL_LEAKAGE_DIVERGED,
            error_floor: REFINED_ERROR_FLOOR,
            error_ceil: REFINED_ERROR_CEIL,
            noise_gate: REFINED_INITIAL_NOISE_GATE,
        }
    }

    fn average(&self, to: &Self, from_weight: f32) -> Self {
        let avg = |a, b| a * from_weight + b * (1.0 - from_weight);
        Self {
            leakage_converged: avg(self.leakage_converged, to.leakage_converged),
            leakage_diverged: avg(self.leakage_diverged, to.leakage_diverged),
            error_floor: avg(self.error_floor, to.error_floor),
            error_ceil: avg(self.error_ceil, to.error_ceil),
            noise_gate: avg(self.noise_gate, to.noise_gate),
        }
    }
}

/// 精滤波器更新增益（`RefinedFilterUpdateGain`）。
pub struct RefinedFilterUpdateGain {
    config_change_duration_blocks: usize,
    one_by_config_change_duration_blocks: f32,
    current_config: RefinedConfig,
    target_config: RefinedConfig,
    old_target_config: RefinedConfig,
    config_change_counter: usize,
    h_error: [f32; FFT_LENGTH_BY_2_PLUS_1],
    poor_excitation_counter: usize,
    call_counter: usize,
}

impl RefinedFilterUpdateGain {
    pub fn new(initial_config: RefinedConfig) -> Self {
        let mut s = Self {
            config_change_duration_blocks: CONFIG_CHANGE_DURATION_BLOCKS,
            one_by_config_change_duration_blocks: 1.0 / CONFIG_CHANGE_DURATION_BLOCKS as f32,
            current_config: initial_config,
            target_config: initial_config,
            old_target_config: initial_config,
            config_change_counter: 0,
            h_error: [H_ERROR_INITIAL; FFT_LENGTH_BY_2_PLUS_1],
            poor_excitation_counter: POOR_EXCITATION_COUNTER_INITIAL,
            call_counter: 0,
        };
        s.set_config(initial_config, true);
        s
    }

    /// 回声路径变化（`HandleEchoPathChange`）。
    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        if variability.delay_change != crate::echo_path_variability::DelayAdjustment::None {
            self.h_error = [H_ERROR_INITIAL; FFT_LENGTH_BY_2_PLUS_1];
        }
        if !variability.gain_change {
            self.poor_excitation_counter = POOR_EXCITATION_COUNTER_INITIAL;
            self.call_counter = 0;
        }
    }

    /// 设置配置；`immediate=true` 立即生效，否则 250 块渐变（`SetConfig`）。
    pub fn set_config(&mut self, config: RefinedConfig, immediate: bool) {
        self.target_config = config;
        if immediate {
            self.current_config = config;
            self.old_target_config = config;
            self.config_change_counter = 0;
        } else {
            self.config_change_counter = self.config_change_duration_blocks;
        }
    }

    /// 配置渐变（`UpdateCurrentConfig`）。
    fn update_current_config(&mut self) {
        if self.config_change_counter > 0 {
            self.config_change_counter -= 1;
            if self.config_change_counter > 0 {
                let change_factor =
                    self.config_change_counter as f32 * self.one_by_config_change_duration_blocks;
                self.current_config = self
                    .old_target_config
                    .average(&self.target_config, change_factor);
            } else {
                self.current_config = self.target_config;
                self.old_target_config = self.target_config;
            }
        }
    }

    /// 计算更新增益 G（`RefinedFilterUpdateGain::Compute`）。
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        subtractor_output: &SubtractorOutput,
        erl: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        size_partitions: usize,
        saturated_capture_signal: bool,
        disallow_leakage_diverged: bool,
        gain_fft: &mut FftData,
    ) {
        let e_refined = &subtractor_output.e_refined_fft;
        let e2_refined = &subtractor_output.e2_refined_spectrum;
        let e2_coarse = &subtractor_output.e2_coarse_spectrum;
        let x2 = render_power;
        self.call_counter += 1;
        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_excitation_counter = 0;
        }

        // 弱激励 / 饱和 / 启动期不更新。
        self.poor_excitation_counter += 1;
        if self.poor_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            gain_fft.re.fill(0.0);
            gain_fft.im.fill(0.0);
        } else {
            // mu = H_err / (0.5·H_err·X2 + P·E2)
            let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                if x2[k] >= self.current_config.noise_gate {
                    mu[k] = self.h_error[k]
                        / (0.5 * self.h_error[k] * x2[k]
                            + size_partitions as f32 * e2_refined[k]);
                }
            }
            render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);
            // H_err −= 0.5·mu·X2·H_err
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.h_error[k] -= 0.5 * mu[k] * x2[k] * self.h_error[k];
            }
            // G = mu·E
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                gain_fft.re[k] = mu[k] * e_refined.re[k];
                gain_fft.im[k] = mu[k] * e_refined.im[k];
            }
        }

        // 泄漏注入 + 钳位。
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if e2_refined[k] <= e2_coarse[k] || disallow_leakage_diverged {
                self.h_error[k] += self.current_config.leakage_converged * erl[k];
            } else {
                self.h_error[k] += self.current_config.leakage_diverged * erl[k];
            }
            self.h_error[k] = self.h_error[k].max(self.current_config.error_floor);
            self.h_error[k] = self.h_error[k].min(self.current_config.error_ceil);
        }
    }

    pub fn h_error(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.h_error
    }
}
