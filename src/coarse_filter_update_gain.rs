//! 粗滤波器更新增益（定步长 NLMS），对照
//! `modules/audio_processing/aec3/coarse_filter_update_gain.{h,cc}`。
//!
//! `mu = rate / X2`（X2 **>** noise_gate 时，注意与 refined 的 >= 不同），
//! `G = mu·E`；弱激励/饱和/启动期不更新。

use crate::constants::{
    CONFIG_CHANGE_DURATION_BLOCKS, COARSE_INITIAL_NOISE_GATE, COARSE_INITIAL_RATE,
    COARSE_NOISE_GATE, COARSE_RATE, FFT_LENGTH_BY_2_PLUS_1,
};
use crate::fft_data::FftData;
use crate::render_signal_analyzer::RenderSignalAnalyzer;

/// 粗滤波器增益配置（`EchoCanceller3Config::Filter::CoarseConfiguration`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarseConfig {
    pub rate: f32,
    pub noise_gate: f32,
}

impl CoarseConfig {
    /// 稳态（`filter.coarse`）。
    pub fn normal() -> Self {
        Self {
            rate: COARSE_RATE,
            noise_gate: COARSE_NOISE_GATE,
        }
    }
    /// 初始（`filter.coarse_initial`）。
    pub fn initial() -> Self {
        Self {
            rate: COARSE_INITIAL_RATE,
            noise_gate: COARSE_INITIAL_NOISE_GATE,
        }
    }
}

/// 粗滤波器更新增益（`CoarseFilterUpdateGain`）。
pub struct CoarseFilterUpdateGain {
    config_change_duration_blocks: usize,
    one_by_config_change_duration_blocks: f32,
    current_config: CoarseConfig,
    target_config: CoarseConfig,
    old_target_config: CoarseConfig,
    config_change_counter: usize,
    poor_signal_excitation_counter: usize,
    call_counter: usize,
}

impl CoarseFilterUpdateGain {
    pub fn new(initial_config: CoarseConfig) -> Self {
        let mut s = Self {
            config_change_duration_blocks: CONFIG_CHANGE_DURATION_BLOCKS,
            one_by_config_change_duration_blocks: 1.0 / CONFIG_CHANGE_DURATION_BLOCKS as f32,
            current_config: initial_config,
            target_config: initial_config,
            old_target_config: initial_config,
            config_change_counter: 0,
            poor_signal_excitation_counter: 0,
            call_counter: 0,
        };
        s.set_config(initial_config, true);
        s
    }

    pub fn handle_echo_path_change(&mut self) {
        self.poor_signal_excitation_counter = 0;
        self.call_counter = 0;
    }

    pub fn set_config(&mut self, config: CoarseConfig, immediate: bool) {
        self.target_config = config;
        if immediate {
            self.current_config = config;
            self.old_target_config = config;
            self.config_change_counter = 0;
        } else {
            self.config_change_counter = self.config_change_duration_blocks;
        }
    }

    fn update_current_config(&mut self) {
        if self.config_change_counter > 0 {
            self.config_change_counter -= 1;
            if self.config_change_counter > 0 {
                let change_factor =
                    self.config_change_counter as f32 * self.one_by_config_change_duration_blocks;
                self.current_config.rate =
                    self.old_target_config.rate * change_factor
                        + self.target_config.rate * (1.0 - change_factor);
                self.current_config.noise_gate =
                    self.old_target_config.noise_gate * change_factor
                        + self.target_config.noise_gate * (1.0 - change_factor);
            } else {
                self.current_config = self.target_config;
                self.old_target_config = self.target_config;
            }
        }
    }

    /// 计算更新增益 G（`CoarseFilterUpdateGain::Compute`）。
    /// `e` 为驱动误差谱：正常用 E_coarse，coarse 刚被重置为 refined 时用 E_refined。
    pub fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        e: &FftData,
        size_partitions: usize,
        saturated_capture_signal: bool,
        gain_fft: &mut FftData,
    ) {
        self.call_counter += 1;
        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_signal_excitation_counter = 0;
        }

        self.poor_signal_excitation_counter += 1;
        if self.poor_signal_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            gain_fft.re.fill(0.0);
            gain_fft.im.fill(0.0);
            return;
        }

        let x2 = render_power;
        let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if x2[k] > self.current_config.noise_gate {
                mu[k] = self.current_config.rate / x2[k];
            }
        }
        render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            gain_fft.re[k] = mu[k] * e.re[k];
            gain_fft.im[k] = mu[k] * e.im[k];
        }
    }
}
