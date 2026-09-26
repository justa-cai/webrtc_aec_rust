//! 抑制增益计算，对照
//! `modules/audio_processing/aec3/suppression_gain.{h,cc}`、
//! `dominant_nearend_detector.{h,cc}`、`moving_average_spectrum.{h,cc}`。
//!
//! 流程：近端主导检测（计数器状态机）→ 低带逐 bin 增益（可闻性加权 →
//! enr/emr 掩蔽增益 → min/max 钳位与速率限制 → 边界处理 → 开方）。
//! G 内部为**功率域**，输出前 `sqrt`；`last_gain_` 保存功率域。
//!
//! 本仓库版本无上游新版的四态状态机/moving_tuning/transient（见 README）。

use crate::constants::{
    BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1, DominantNearendDetectionConfig,
    ECHO_AUDIBILITY_FLOOR_POWER, ECHO_AUDIBILITY_LOW_RENDER_LIMIT,
    ECHO_AUDIBILITY_NORMAL_RENDER_LIMIT, ECHO_AUDIBILITY_THRESHOLD,
    LOW_NOISE_RENDER_PEAK_FACTOR, LOW_NOISE_RENDER_THRESHOLD, SuppressorConfig,
    SuppressorTuning,
};

// ---------------------------------------------------------------------------
// MovingAverageSpectrum（moving_average_spectrum.cc）
// ---------------------------------------------------------------------------

/// 近端谱的 boxcar 滑动平均（`MovingAverageSpectrum`）。
struct MovingAverageSpectrum {
    memory: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    mem_index: usize,
    /// 实际记忆长度（= N−1 条历史 + 当前输入）。
    mem_len: usize,
    number_updates: usize,
}

impl MovingAverageSpectrum {
    fn new(average_length: usize) -> Self {
        let mem_len = average_length.saturating_sub(1).max(1);
        Self {
            memory: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; mem_len],
            mem_index: 0,
            mem_len,
            number_updates: 0,
        }
    }

    fn average(&mut self, x: &[f32; FFT_LENGTH_BY_2_PLUS_1]) -> [f32; FFT_LENGTH_BY_2_PLUS_1] {
        let mut out = *x;
        for m in &self.memory {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                out[k] += m[k];
            }
        }
        let denom = (self.number_updates + 1) as f32;
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            out[k] /= denom;
        }
        self.memory[self.mem_index] = *x;
        self.mem_index = (self.mem_index + 1) % self.mem_len;
        self.number_updates = (self.number_updates + 1).min(self.mem_len);
        out
    }
}

// ---------------------------------------------------------------------------
// DominantNearendDetector（dominant_nearend_detector.cc）
// ---------------------------------------------------------------------------

/// 低频段求和：bins 1..=15（跳过 DC，`LOW_FREQ_SUM`）。
fn low_freq_sum(s: &[f32; FFT_LENGTH_BY_2_PLUS_1]) -> f32 {
    s[1..16].iter().sum()
}

/// 近端主导检测器（`DominantNearendDetector`）。单声道。
struct DominantNearendDetector {
    cfg: DominantNearendDetectionConfig,
    trigger_counter: i32,
    hold_counter: i32,
    nearend_state: bool,
}

impl DominantNearendDetector {
    fn new(cfg: DominantNearendDetectionConfig) -> Self {
        Self {
            cfg,
            trigger_counter: 0,
            hold_counter: 0,
            nearend_state: false,
        }
    }

    fn is_nearend_state(&self) -> bool {
        self.nearend_state
    }

    /// 每块更新（`DominantNearendDetector::Update`）。
    fn update(
        &mut self,
        nearend_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        residual_echo_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        comfort_noise_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        initial_state: bool,
    ) {
        let ne_sum = low_freq_sum(nearend_spectrum);
        let echo_sum = low_freq_sum(residual_echo_spectrum);
        let noise_sum = low_freq_sum(comfort_noise_spectrum);

        // 触发：回声明显小于近端且近端高于噪声。
        if (!initial_state || self.cfg.use_during_initial_phase)
            && echo_sum < self.cfg.enr_threshold * ne_sum
            && ne_sum > self.cfg.snr_threshold * noise_sum
        {
            self.trigger_counter += 1;
            if self.trigger_counter >= self.cfg.trigger_threshold {
                self.hold_counter = self.cfg.hold_duration;
                // 计数器饱和在阈值（保持触发条件持续满足的状态）。
                self.trigger_counter = self.cfg.trigger_threshold;
            }
        } else {
            self.trigger_counter = (self.trigger_counter - 1).max(0);
        }

        // 强回声提前退出。
        if echo_sum > self.cfg.enr_exit_threshold * ne_sum
            && echo_sum > self.cfg.snr_threshold * noise_sum
        {
            self.hold_counter = 0;
        }

        self.hold_counter = (self.hold_counter - 1).max(0);
        self.nearend_state = self.hold_counter > 0;
    }
}

// ---------------------------------------------------------------------------
// LowNoiseRenderDetector（suppression_gain.cc:461-478）
// ---------------------------------------------------------------------------

/// 低噪声渲染检测器（`LowNoiseRenderDetector`）。
struct LowNoiseRenderDetector {
    average_power: f32,
}

impl LowNoiseRenderDetector {
    fn new() -> Self {
        Self {
            average_power: 32768.0 * 32768.0,
        }
    }

    /// 先判后更新（`Detect`）。
    fn detect(&mut self, render: &[f32; BLOCK_SIZE]) -> bool {
        let mut x2_sum: f32 = 0.0;
        let mut x2_max = 0.0f32;
        for v in render.iter() {
            let p = v * v;
            x2_sum += p;
            x2_max = x2_max.max(p);
        }
        let low_noise_render = self.average_power < LOW_NOISE_RENDER_THRESHOLD
            && x2_max < LOW_NOISE_RENDER_PEAK_FACTOR * self.average_power;
        self.average_power = 0.9 * self.average_power + 0.1 * x2_sum;
        low_noise_render
    }
}

// ---------------------------------------------------------------------------
// GainParameters（suppression_gain.cc:487-512）
// ---------------------------------------------------------------------------

/// 逐 bin 掩蔽阈值参数（LF/HF 两组线性内插）。
struct GainParameters {
    enr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
    enr_suppress: [f32; FFT_LENGTH_BY_2_PLUS_1],
    emr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
    max_inc_factor: f32,
    max_dec_factor_lf: f32,
}

impl GainParameters {
    fn set_config(last_lf_band: usize, first_hf_band: usize, tuning: &SuppressorTuning) -> Self {
        let mut p = Self {
            enr_transparent: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            enr_suppress: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            emr_transparent: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            max_inc_factor: tuning.max_inc_factor,
            max_dec_factor_lf: tuning.max_dec_factor_lf,
        };
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let a = if k <= last_lf_band {
                0.0
            } else if k < first_hf_band {
                (k - last_lf_band) as f32 / (first_hf_band - last_lf_band) as f32
            } else {
                1.0
            };
            p.enr_transparent[k] = (1.0 - a) * tuning.lf.enr_transparent
                + a * tuning.hf.enr_transparent;
            p.enr_suppress[k] =
                (1.0 - a) * tuning.lf.enr_suppress + a * tuning.hf.enr_suppress;
            p.emr_transparent[k] = (1.0 - a) * tuning.lf.emr_transparent
                + a * tuning.hf.emr_transparent;
        }
        p
    }
}

// ---------------------------------------------------------------------------
// 增益函数（suppression_gain.cc 内部函数）
// ---------------------------------------------------------------------------

/// 回声可闻性加权（`WeightEchoForAudibility`，:88-121）。
/// 三段 [0,3)/[3,7)/[7,65)，低能量回声按可闻性打折。
fn weight_echo_for_audibility(echo: &[f32; FFT_LENGTH_BY_2_PLUS_1]) -> [f32; FFT_LENGTH_BY_2_PLUS_1] {
    let mut wre = *echo;
    let thresholds = [
        ECHO_AUDIBILITY_FLOOR_POWER * ECHO_AUDIBILITY_THRESHOLD, // lf
        ECHO_AUDIBILITY_FLOOR_POWER * ECHO_AUDIBILITY_THRESHOLD, // mf
        ECHO_AUDIBILITY_FLOOR_POWER * ECHO_AUDIBILITY_THRESHOLD, // hf
    ];
    for k in 0..3 {
        if wre[k] < thresholds[0] {
            let t = (thresholds[0] - wre[k]) / (thresholds[0] - ECHO_AUDIBILITY_FLOOR_POWER);
            wre[k] *= (1.0 - t * t).max(0.0);
        }
    }
    for k in 3..7 {
        if wre[k] < thresholds[1] {
            let t = (thresholds[1] - wre[k]) / (thresholds[1] - ECHO_AUDIBILITY_FLOOR_POWER);
            wre[k] *= (1.0 - t * t).max(0.0);
        }
    }
    for k in 7..FFT_LENGTH_BY_2_PLUS_1 {
        if wre[k] < thresholds[2] {
            let t = (thresholds[2] - wre[k]) / (thresholds[2] - ECHO_AUDIBILITY_FLOOR_POWER);
            wre[k] *= (1.0 - t * t).max(0.0);
        }
    }
    wre
}

/// 最大增益（上升限速，`GetMaxGain`，:280-289）。
fn get_max_gain(
    last_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    inc_factor: f32,
    floor_first_increase: f32,
    max_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        max_gain[k] = ((last_gain[k] * inc_factor).max(floor_first_increase)).min(1.0);
    }
}

/// 掩蔽增益（`GainToNoAudibleEcho`，:215-233）——核心逐 bin 抑制函数。
fn gain_to_no_audible_echo(
    nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    masker: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    params: &GainParameters,
    gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        let enr = echo[k] / (nearend[k] + 1.0);
        let emr = echo[k] / (masker[k] + 1.0);
        let mut g = 1.0;
        if enr > params.enr_transparent[k] && emr > params.emr_transparent[k] {
            g = (params.enr_suppress[k] - enr)
                / (params.enr_suppress[k] - params.enr_transparent[k]);
            g = g.max(params.emr_transparent[k] / emr);
        }
        gain[k] = g;
    }
}

/// 低频边界（`LimitLowFrequencyGains`，:38-42）。
fn limit_low_frequency_gains(gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    gain[1] = gain[1].min(gain[2]);
    gain[0] = gain[1];
}

/// 高频限制（`LimitHighFrequencyGains`，:44-85）。
fn limit_high_frequency_gains(
    cfg: &SuppressorConfig,
    conservative: bool,
    gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let band = cfg.high_frequency_suppression.limiting_gain_band;
    let n = cfg.high_frequency_suppression.bands_in_limiting_gain;
    let min_upper_gain = gain[band..(band + n).min(FFT_LENGTH_BY_2_PLUS_1)]
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    for k in (band + 1)..FFT_LENGTH_BY_2_PLUS_1 {
        gain[k] = gain[k].min(min_upper_gain);
    }
    gain[FFT_LENGTH_BY_2_PLUS_1 - 1] = gain[FFT_LENGTH_BY_2_PLUS_1 - 2];
    if conservative {
        let hf_bound: f32 = gain[20..29].iter().sum::<f32>() / 9.0;
        for k in 29..FFT_LENGTH_BY_2_PLUS_1 {
            gain[k] = gain[k].min(hf_bound);
        }
    }
}

/// 高带增益（`UpperBandsGain`，:127-212）。16 kHz 单带 → 恒 1.0
/// （上游 `render.NumBands() == 1` 分支），多带逻辑见 README。
fn upper_bands_gain() -> f32 {
    1.0
}

// ---------------------------------------------------------------------------
// SuppressionGain 主结构
// ---------------------------------------------------------------------------

/// 抑制增益计算器（`SuppressionGain`）。单声道。
pub struct SuppressionGain {
    nearend_smoother: MovingAverageSpectrum,
    detector: DominantNearendDetector,
    low_render_detector: LowNoiseRenderDetector,
    normal_parameters: GainParameters,
    nearend_parameters: GainParameters,
    last_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    last_nearend: [f32; FFT_LENGTH_BY_2_PLUS_1],
    last_echo: [f32; FFT_LENGTH_BY_2_PLUS_1],
    initial_state: bool,
    // 复用缓冲
    max_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    min_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for SuppressionGain {
    fn default() -> Self {
        Self::new()
    }
}

impl SuppressionGain {
    pub fn new() -> Self {
        let cfg = SuppressorConfig::default();
        Self::with_config(&cfg)
    }

    pub fn with_config(cfg: &SuppressorConfig) -> Self {
        Self {
            nearend_smoother: MovingAverageSpectrum::new(cfg.nearend_average_blocks),
            detector: DominantNearendDetector::new(cfg.dominant_nearend_detection),
            low_render_detector: LowNoiseRenderDetector::new(),
            normal_parameters: GainParameters::set_config(
                cfg.last_lf_band,
                cfg.first_hf_band,
                &cfg.normal_tuning,
            ),
            nearend_parameters: GainParameters::set_config(
                cfg.last_lf_band,
                cfg.first_hf_band,
                &cfg.nearend_tuning,
            ),
            last_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            last_nearend: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            last_echo: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            initial_state: true,
            max_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            min_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    pub fn set_initial_state(&mut self, state: bool) {
        self.initial_state = state;
    }

    /// 近端是否主导（`IsDominantNearend`；注意 EchoRemover 在 GetGain 之前
    /// 读取时拿到的是**上一块**的状态——与源码顺序语义一致）。
    pub fn is_dominant_nearend(&self) -> bool {
        self.detector.is_nearend_state()
    }

    /// 每块计算增益（`SuppressionGain::GetGain`）。`gain` 为输出（幅度域，
    /// 内部先在功率域计算再开方）；返回高带增益（16k 单带恒 1.0）。
    #[allow(clippy::too_many_arguments)]
    pub fn get_gain(
        &mut self,
        cfg: &SuppressorConfig,
        nearend_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        r2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        r2_unbounded: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        n2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        saturated_echo: bool,
        render_block: &[f32; BLOCK_SIZE],
        clock_drift: bool,
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) -> f32 {
        // 检测用回声谱（默认无界）。
        let echo_for_detection = if cfg.dominant_nearend_detection.use_unbounded_echo_spectrum {
            r2_unbounded
        } else {
            r2
        };
        self.detector.update(
            nearend_spectrum,
            echo_for_detection,
            n2,
            self.initial_state,
        );
        let nearend_state = self.detector.is_nearend_state();

        let low_noise_render = self.low_render_detector.detect(render_block);

        // —— LowerBandGain（:291-348，单声道展开）——
        gain.fill(1.0);
        let inc_factor = if nearend_state {
            self.nearend_parameters.max_inc_factor
        } else {
            self.normal_parameters.max_inc_factor
        };
        get_max_gain(&self.last_gain, inc_factor, cfg.floor_first_increase, &mut self.max_gain);

        let nearend = self.nearend_smoother.average(nearend_spectrum);
        let wre = weight_echo_for_audibility(r2);
        let params = if nearend_state {
            &self.nearend_parameters
        } else {
            &self.normal_parameters
        };

        // GetMinGain（含低频下降限速，需 last_gain——合并进 min_gain 计算）。
        if saturated_echo {
            self.min_gain.fill(0.0);
        } else {
            let min_echo_power = if low_noise_render {
                ECHO_AUDIBILITY_LOW_RENDER_LIMIT
            } else {
                ECHO_AUDIBILITY_NORMAL_RENDER_LIMIT
            };
            let dec_factor_lf = params.max_dec_factor_lf;
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.min_gain[k] = if wre[k] > 0.0 {
                    (min_echo_power / wre[k]).min(1.0)
                } else {
                    1.0
                };
            }
            if !self.initial_state || cfg.lf_smoothing_during_initial_phase {
                for k in 0..=cfg.last_lf_smoothing_band {
                    if k <= cfg.last_permanent_lf_smoothing_band
                        || self.last_nearend[k] > self.last_echo[k]
                    {
                        self.min_gain[k] =
                            self.min_gain[k].max(self.last_gain[k] * dec_factor_lf).min(1.0);
                    }
                }
            }
        }

        // GainToNoAudibleEcho（masker = N2；实验开关下用 max(N2, 平滑近端谱)）。
        let mut g_ch = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        if cfg.nearend_masker {
            let mut masker = *n2;
            let alpha = cfg.nearend_masker_alpha.clamp(0.0, 1.0);
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                masker[k] = masker[k].max(alpha * nearend[k]);
            }
            gain_to_no_audible_echo(&nearend, &wre, &masker, params, &mut g_ch);
        } else {
            gain_to_no_audible_echo(&nearend, &wre, n2, params, &mut g_ch);
        }
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            g_ch[k] = g_ch[k].min(self.max_gain[k]).max(self.min_gain[k]);
            gain[k] = gain[k].min(g_ch[k]);
        }
        self.last_nearend = nearend;
        self.last_echo = wre;

        limit_low_frequency_gains(gain);
        if !nearend_state || clock_drift || cfg.conservative_hf_suppression {
            limit_high_frequency_gains(cfg, cfg.conservative_hf_suppression, gain);
        }

        // 功率域保存 last_gain，输出转幅度域。
        self.last_gain = *gain;
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            gain[k] = gain[k].sqrt();
        }

        upper_bands_gain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 检测器状态机：触发（12 块）→ 保持（50 块）→ 提前退出。
    #[test]
    fn detector_state_machine() {
        let cfg = SuppressorConfig::default();
        let mut det = DominantNearendDetector::new(cfg.dominant_nearend_detection);
        let ne = [1000.0f32; 65]; // 近端强
        let echo = [100.0f32; 65]; // 回声弱（echo/ne=0.1 < 0.25）
        let noise = [1.0f32; 65]; // ne/noise=1000 > 30
        for i in 0..11 {
            det.update(&ne, &echo, &noise, false);
            assert!(!det.is_nearend_state(), "i={} 不应提前触发", i);
        }
        det.update(&ne, &echo, &noise, false); // 第 12 块
        assert!(det.is_nearend_state(), "12 块应触发");
        // 近端停止（触发条件不满足）→ hold 期间仍为真。
        // 注意触发块内 hold 已同步递减（50→49），故保持 49 块。
        let ne_off = [1.0f32; 65];
        let echo_off = [1.0f32; 65];
        for i in 0..48 {
            det.update(&ne_off, &echo_off, &noise, false);
            assert!(det.is_nearend_state(), "hold 内 i={} 应保持", i);
        }
        det.update(&ne_off, &echo_off, &noise, false); // i=48：hold 减到 0
        assert!(!det.is_nearend_state(), "hold 结束应退出");
        // 强回声提前退出
        for _ in 0..12 {
            det.update(&ne, &echo, &noise, false);
        }
        assert!(det.is_nearend_state());
        det.update(&[100.0; 65], &[10000.0; 65], &[1.0; 65], false); // echo/ne=100 > 10
        assert!(!det.is_nearend_state(), "强回声应立即退出");
    }

    /// 增益单调性：回声越强增益越低；近端主导时增益更高（保护近端）。
    #[test]
    fn gain_monotonic_in_echo() {
        let cfg = SuppressorConfig::default();
        let n2 = [1000.0f32; 65];
        let ne = [10000.0f32; 65];
        let mut sg = SuppressionGain::new();
        let mut g1 = [0.0f32; 65];
        let mut g2 = [0.0f32; 65];
        let render = [3000.0f32; BLOCK_SIZE];
        sg.get_gain(&cfg, &ne, &[1000.0; 65], &[1000.0; 65], &n2, false, &render, false, &mut g1);
        sg = SuppressionGain::new();
        sg.get_gain(&cfg, &ne, &[100000.0; 65], &[100000.0; 65], &n2, false, &render, false, &mut g2);
        for k in [16usize, 32] {
            assert!(g2[k] <= g1[k], "更强回声应更低增益: k={} {} vs {}", k, g2[k], g1[k]);
        }
    }

    /// 全静默（零回声零近端）：增益应为 1（不抑制）。
    #[test]
    fn silence_passes() {
        let cfg = SuppressorConfig::default();
        let mut sg = SuppressionGain::new();
        let mut g = [0.0f32; 65];
        let render = [0.0f32; BLOCK_SIZE];
        for _ in 0..50 {
            sg.get_gain(&cfg, &[1.0; 65], &[0.0; 65], &[0.0; 65], &[10.0; 65], false, &render, false, &mut g);
        }
        for k in [16usize, 48] {
            assert!(g[k] > 0.99, "k={} g={} 静默不应抑制", k, g[k]);
        }
    }

    /// 速率限制：增益骤降请求被 min_gain（0.25·last）托底，逐块下降。
    #[test]
    fn descent_rate_limited() {
        let cfg = SuppressorConfig::default();
        let mut sg = SuppressionGain::new();
        let mut g = [0.0f32; 65];
        let render = [3000.0f32; BLOCK_SIZE];
        // 先静默若干块（增益=1）
        for _ in 0..10 {
            sg.get_gain(&cfg, &[1.0; 65], &[0.0; 65], &[0.0; 65], &[10.0; 65], false, &render, false, &mut g);
        }
        // 突然强回声：功率域增益不能一步降到底（低频尤其被 0.25·last 托底）
        let ne = [100.0; 65];
        let echo = [1e7; 65];
        sg.get_gain(&cfg, &ne, &echo, &echo, &[10.0; 65], false, &render, false, &mut g);
        // 幅度域 = sqrt(功率域)。低频 bin2 的功率域下限 = max(min_echo/wre, 0.25·last_gain=0.25)
        // → 幅度 ≥ sqrt(0.25) = 0.5
        assert!(
            g[2] >= 0.49,
            "低频下降限速失效: g[2]={}",
            g[2]
        );
    }
}
