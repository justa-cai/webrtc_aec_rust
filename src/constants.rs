//! 全部固定常量，对照 `modules/audio_processing/aec3/aec3_common.h` 与
//! `api/audio/echo_canceller3_config.h`（默认值）。
//!
//! 本实现单声道、16 kHz，因此只保留线性路径实际消费的子集；
//! 多声道/高带/抑制器相关常量未移植。

// ---------------------------------------------------------------------------
// aec3_common.h
// ---------------------------------------------------------------------------

/// FFT 长度的一半 = 64（`kFftLengthBy2`）。
pub const FFT_LENGTH_BY_2: usize = 64;
/// 有效 bin 数 = 65（`kFftLengthBy2Plus1`，DC..Nyquist 含端点）。
pub const FFT_LENGTH_BY_2_PLUS_1: usize = FFT_LENGTH_BY_2 + 1;
/// FFT 长度 = 128（`kFftLength`）。
pub const FFT_LENGTH: usize = 2 * FFT_LENGTH_BY_2;
/// `kBlockSizeLog2`：样本↔块换算的移位数。
pub const BLOCK_SIZE_LOG2: usize = 6;
/// 块大小 = 64 样本 @16 kHz = 4 ms（`kBlockSize`）。
pub const BLOCK_SIZE: usize = FFT_LENGTH_BY_2;
/// 10 ms 帧长 = 160 样本（`kFrameSize`）。
pub const FRAME_SIZE: usize = 160;
/// 每秒块数（16 kHz / 64）（`kNumBlocksPerSecond`）。
pub const NUM_BLOCKS_PER_SECOND: usize = 250;
/// 匹配滤波器窗口长度（子块数，`kMatchedFilterWindowSizeSubBlocks`）。
pub const MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS: usize = 32;
/// 相邻匹配滤波器错开的子块步进（`kMatchedFilterAlignmentShiftSizeSubBlocks` = 32*3/4）。
pub const MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS: usize =
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS * 3 / 4;

/// 时域长度换算（`GetTimeDomainLength`）：块数 × 64。
pub const fn time_domain_length(filter_length_blocks: usize) -> usize {
    filter_length_blocks * FFT_LENGTH_BY_2
}

// ---------------------------------------------------------------------------
// echo_canceller3_config.h —— Filter（滤波器默认配置）
// ---------------------------------------------------------------------------

/// refined 稳态滤波器长度（块）：13 块 = 832 抽头 ≈ 52 ms。
pub const FILTER_REFINED_LENGTH_BLOCKS: usize = 13;
/// refined 初始（未收敛期）滤波器长度：12 块。
pub const FILTER_REFINED_INITIAL_LENGTH_BLOCKS: usize = 12;
/// coarse 稳态滤波器长度：13 块。
pub const FILTER_COARSE_LENGTH_BLOCKS: usize = 13;
/// coarse 初始滤波器长度：12 块。
pub const FILTER_COARSE_INITIAL_LENGTH_BLOCKS: usize = 12;

/// refined 收敛后的频域泄漏系数（`leakage_converged`）。
pub const REFINED_LEAKAGE_CONVERGED: f32 = 0.00005;
/// refined 判定发散时的频域泄漏系数（`leakage_diverged`）。
pub const REFINED_LEAKAGE_DIVERGED: f32 = 0.05;
/// refined 初始阶段的收敛泄漏（`refined_initial.leakage_converged`）。
pub const REFINED_INITIAL_LEAKAGE_CONVERGED: f32 = 0.005;
/// refined 初始阶段的发散泄漏（`refined_initial.leakage_diverged`）。
pub const REFINED_INITIAL_LEAKAGE_DIVERGED: f32 = 0.5;
/// `H_error` 下限（`error_floor`）。
pub const REFINED_ERROR_FLOOR: f32 = 0.001;
/// `H_error` 上限（`error_ceil`）。
pub const REFINED_ERROR_CEIL: f32 = 2.0;
/// 每 bin 更新门限（`noise_gate`，对应约 −39 dBFS 的 WGN 功率）。
pub const REFINED_NOISE_GATE: f32 = 20075344.0;
/// refined 初始阶段噪声门限（与稳态相同）。
pub const REFINED_INITIAL_NOISE_GATE: f32 = 20075344.0;

/// coarse 稳态 NLMS 步长（`rate`）。
pub const COARSE_RATE: f32 = 0.7;
/// coarse 初始阶段步长。
pub const COARSE_INITIAL_RATE: f32 = 0.9;
/// coarse 噪声门限（与 refined 相同）。
pub const COARSE_NOISE_GATE: f32 = 20075344.0;
/// coarse 初始阶段噪声门限。
pub const COARSE_INITIAL_NOISE_GATE: f32 = 20075344.0;

/// 长度/配置参数渐变时长（块）= 1 s（`config_change_duration_blocks`）。
pub const CONFIG_CHANGE_DURATION_BLOCKS: usize = 250;
/// 初始状态时长（秒，`initial_state_seconds` → 2.5*250 = 625 块后退出初始状态）。
pub const INITIAL_STATE_BLOCKS: usize = 625;
/// coarse 被重置为 refined 后的 hangover 块数（`coarse_reset_hangover_blocks`）。
pub const COARSE_RESET_HANGOVER_BLOCKS: usize = 25;

// ---------------------------------------------------------------------------
// echo_canceller3_config.h —— Delay（时延估计默认配置）
// ---------------------------------------------------------------------------

/// 时延估计降采样倍数（`down_sampling_factor`）。
pub const DELAY_DOWN_SAMPLING_FACTOR: usize = 4;
/// 降采样域子块长度 = 64/4 = 16 样本。
pub const DELAY_SUB_BLOCK_SIZE: usize = BLOCK_SIZE / DELAY_DOWN_SAMPLING_FACTOR;
/// 并行匹配滤波器个数（`num_filters`）。
pub const DELAY_NUM_FILTERS: usize = 5;
/// 单个匹配滤波器长度（降采样样本）= 32 子块 × 16。
pub const MATCHED_FILTER_SIZE: usize =
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS * DELAY_SUB_BLOCK_SIZE; // 512
/// 相邻匹配滤波器错开（降采样样本）= 24 子块 × 16。
pub const MATCHED_FILTER_INTRA_LAG_SHIFT: usize =
    MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS * DELAY_SUB_BLOCK_SIZE; // 384
/// 报告延迟时扣除的保护余量（全带样本，`delay_headroom_samples`）。
pub const DELAY_HEADROOM_SAMPLES: usize = 32;
/// 缓冲迟滞块数（`hysteresis_limit_blocks`，仅 refined↔refined 时启用）。
pub const DELAY_HYSTERESIS_LIMIT_BLOCKS: usize = 1;
/// 匹配滤波器 NLMS 步长——搜索阶段（`delay_estimate_smoothing`）。
pub const DELAY_ESTIMATE_SMOOTHING_FAST: f32 = 0.7;
/// 匹配滤波器 NLMS 步长——锁定后（`delay_estimate_smoothing_delay_found`）。
pub const DELAY_ESTIMATE_SMOOTHING_SLOW: f32 = 0.7;
/// 候选可靠性门限：残差须低于基线的该比例（`delay_candidate_detection_threshold`）。
pub const DELAY_CANDIDATE_DETECTION_THRESHOLD: f32 = 0.2;
/// 直方图双门限（`delay_selection_thresholds`）：>5 出 coarse 候选，>20 出 refined。
pub const DELAY_SELECTION_THRESHOLD_INITIAL: i32 = 5;
pub const DELAY_SELECTION_THRESHOLD_CONVERGED: i32 = 20;
/// 无外部时延时的初始缓冲 total delay（块，`default_delay`）。
pub const DELAY_DEFAULT_BLOCKS: usize = 5;
/// 弱激励门限（`poor_excitation_render_limit`，控制匹配滤波更新门限 512·150²）。
pub const POOR_EXCITATION_RENDER_LIMIT: f32 = 150.0;
/// 是否启用前回声检测（`detect_pre_echo`）。
pub const DELAY_DETECT_PRE_ECHO: bool = true;
/// 时钟漂移判定后恢复无漂移所需的稳定块数（clockdrift_detector.cc，7500 块 = 30 s）。
pub const CLOCKDRIFT_STABLE_BLOCKS: i32 = 7500;

// ---------------------------------------------------------------------------
// render_levels / 各处能量门限（换算成块级能量）
// ---------------------------------------------------------------------------

/// 活跃 render 判定门限（`active_render_limit = 100` → 块能量 100²·64）。
pub const ACTIVE_RENDER_LIMIT: f32 = 100.0;
pub const ACTIVE_RENDER_BLOCK_ENERGY: f32 = ACTIVE_RENDER_LIMIT * ACTIVE_RENDER_LIMIT
    * BLOCK_SIZE as f32; // 640000

/// 麦克风帧饱和门限（echo_canceller3.cc `DetectSaturation`：|y| ≥ 32700）。
pub const CAPTURE_SATURATION_LIMIT: f32 = 32700.0;
/// 匹配滤波器逐样本饱和保护门限（matched_filter.cc：|y[i]| ≥ 32000 跳过更新）。
pub const MATCHED_FILTER_SATURATION_LIMIT: f32 = 32000.0;

// ---------------------------------------------------------------------------
// render_delay_buffer 尺寸推导（aec3_common.h GetDownSampledBufferSize /
// GetRenderDelayBufferSize）
// ---------------------------------------------------------------------------

/// 降采样环形缓冲长度（`GetDownSampledBufferSize(4, 5)`）。
pub const DOWNSAMPLED_BUFFER_SIZE: usize =
    BLOCK_SIZE / DELAY_DOWN_SAMPLING_FACTOR
        * (MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS * DELAY_NUM_FILTERS
            + MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS
            + 1); // 16 * (24*5 + 32 + 1) = 16*153 = 2448
/// 块/频谱/FFT 三环形缓冲长度（`GetRenderDelayBufferSize(4, 5, 13)`）。
pub const RENDER_DELAY_BUFFER_SIZE: usize =
    DOWNSAMPLED_BUFFER_SIZE / (BLOCK_SIZE / DELAY_DOWN_SAMPLING_FACTOR)
        + FILTER_REFINED_LENGTH_BLOCKS
        + 1; // 153 + 13 + 1 = 167

// ---------------------------------------------------------------------------
// echo_canceller3_config.h —— Buffering（render_delay_buffer.cc 超量检测用）
// ---------------------------------------------------------------------------

/// 超量 render 检测的间隔（块，`excess_render_detection_interval_blocks`）。
pub const EXCESS_RENDER_DETECTION_INTERVAL_BLOCKS: usize = 250;
/// 允许的最小缓冲余量上限（块，`max_allowed_excess_render_blocks`）。
pub const MAX_ALLOWED_EXCESS_RENDER_BLOCKS: usize = 8;

// ---------------------------------------------------------------------------
// aec_state / filter_analyzer / subtractor_output_analyzer 门限
// ---------------------------------------------------------------------------

/// refined 收敛判定：e2_refined < 0.5·y2 且 y2 > 50²·64。
pub const CONVERGENCE_THRESHOLD: f32 = 50.0 * 50.0 * BLOCK_SIZE as f32; // 160000
/// 低电平时 coarse 宽松收敛门限：20²·64。
pub const CONVERGENCE_THRESHOLD_LOW_LEVEL: f32 = 20.0 * 20.0 * BLOCK_SIZE as f32; // 25600
/// 发散判定门限：30²·64。
pub const DIVERGENCE_THRESHOLD: f32 = 30.0 * 30.0 * BLOCK_SIZE as f32; // 57600

/// FilterAnalyzer 对冲激响应做高通预滤波的抽头（filter_analyzer.cc，≈600 Hz 最小相位）。
pub const FILTER_ANALYZER_HPF_TAPS: [f32; 3] =
    [0.7929742, -0.36072128, -0.47047766];
/// FilterAnalyzer 每块推进的分析区域长度（抽头）。
pub const FILTER_ANALYZER_REGION_SIZE: usize = FFT_LENGTH_BY_2; // 64

// ---------------------------------------------------------------------------
// matched_filter / 聚合器内部常量（matched_filter.cc / matched_filter_lag_aggregator.cc）
// ---------------------------------------------------------------------------

/// 前回声部分误差的子采样率：每 4 个抽头一组（matched_filter.cc）。
pub const ACCUMULATED_ERROR_SUB_SAMPLE_RATE: usize = 4;
/// 前回声起点扫描阈值（`kPreEchoThreshold`）。
pub const PRE_ECHO_THRESHOLD: f32 = 0.5;
/// accumulated_error 平滑：上升系数（`kSmoothConstantIncreases`，快降慢升）。
pub const ACCUMULATED_ERROR_SMOOTH_UP: f32 = 0.015;
/// 前回声估计启用所需的最少更新次数（matched_filter.cc）。
pub const PRE_ECHO_MIN_UPDATES: i32 = 50;
/// 峰值可靠所需的边界余量：lag > 2 且 < size−10（matched_filter.cc）。
pub const MATCHED_FILTER_PEAK_LOW_MARGIN: usize = 2;
pub const MATCHED_FILTER_PEAK_HIGH_MARGIN: usize = 10;

/// 直方图滑动窗口长度（`HighestPeakAggregator::histogram_data_` = 250 项，1 s）。
pub const LAG_HISTOGRAM_WINDOW: usize = 250;
/// 前回声直方图首个 250 项窗口内的组惩罚系数（0.7/组）。
pub const PRE_ECHO_PENALIZATION: f32 = 0.7;
/// 前回声直方图惩罚阶段时长（更新次数 = 2 s）。
pub const PRE_ECHO_PENALIZATION_UPDATES: i32 = 500;

// ---------------------------------------------------------------------------
// echo_remover 线性部分门限（echo_remover.cc）
// ---------------------------------------------------------------------------

/// UseRefinedOutput：麦克风能量门限 30²·64。
pub const USE_REFINED_Y2_THRESHOLD: f32 = 30.0 * 30.0 * BLOCK_SIZE as f32;
/// UseRefinedOutput：回声估计能量门限 60²·64。
pub const USE_REFINED_S2_THRESHOLD: f32 = 60.0 * 60.0 * BLOCK_SIZE as f32;
/// UseRefinedOutput：coarse 明显更优的比例（e2_coarse < 0.9·e2_refined）。
pub const USE_REFINED_COARSE_RATIO: f32 = 0.9;
/// 两级输出切换的交叉淡化长度（样本，`kTransitionSize`）。
pub const SIGNAL_TRANSITION_SIZE: usize = 30;

// ---------------------------------------------------------------------------
// subtractor 失配修正门限（subtractor.cc FilterMisadjustmentEstimator）
// ---------------------------------------------------------------------------

/// 失配估计累积块数。
pub const MISADJUSTMENT_ACCUM_BLOCKS: i32 = 4;
/// 失配估计更新门限：y2 累积 > 4·200²·64。
pub const MISADJUSTMENT_Y2_GATE: f32 = 200.0 * 200.0 * BLOCK_SIZE as f32;
/// overhang 触发门限：e2 累积 > 4·7500²·64。
pub const MISADJUSTMENT_E2_OVERHANG_GATE: f32 = 7500.0 * 7500.0 * BLOCK_SIZE as f32;
/// 触发整体缩放的判定门限：inv_misadjustment > 10。
pub const MISADJUSTMENT_ADJUST_THRESHOLD: f32 = 10.0;
/// coarse 连续表现差于 refined 的判定块数。
pub const POOR_COARSE_FILTER_COUNTER_LIMIT: usize = 5;
/// e_refined 输出钳位（subtractor.cc 末尾 SafeClamp）。
pub const OUTPUT_CLAMP_LIMIT: f32 = 32768.0;

// ===========================================================================
// 非线性层（NLP）常量与配置 —— 对照 echo_canceller3_config.h 与各 .cc 文件
// ===========================================================================

/// 掩蔽阈值组（`MaskingThresholds`，echo_canceller3_config.h:206-212）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskingThresholds {
    pub enr_transparent: f32,
    pub enr_suppress: f32,
    pub emr_transparent: f32,
}

/// 增益调参（`Suppressor::Tuning`，echo_canceller3_config.h:213-221）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuppressorTuning {
    pub lf: MaskingThresholds,
    pub hf: MaskingThresholds,
    /// 增益上升限（功率域倍率/块）。
    pub max_inc_factor: f32,
    /// 低频增益下降限（功率域倍率/块）。
    pub max_dec_factor_lf: f32,
}

/// 近端主导检测配置（`dominant_nearend_detection`，echo_canceller3_config.h:228-237）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DominantNearendDetectionConfig {
    pub enr_threshold: f32,
    pub enr_exit_threshold: f32,
    pub snr_threshold: f32,
    pub hold_duration: i32,
    pub trigger_threshold: i32,
    pub use_during_initial_phase: bool,
    pub use_unbounded_echo_spectrum: bool,
}

/// 高带抑制配置（`high_bands_suppression`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HighBandsSuppressionConfig {
    pub enr_threshold: f32,
    pub max_gain_during_echo: f32,
    pub anti_howling_activation_threshold: f32,
    pub anti_howling_gain: f32,
}

/// 高频限制配置（`high_frequency_suppression`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HighFrequencySuppressionConfig {
    pub limiting_gain_band: usize,
    pub bands_in_limiting_gain: usize,
}

/// 完整抑制器配置（`EchoCanceller3Config::Suppressor`）。
/// ML 注入点 `adjust_config` 以此为参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuppressorConfig {
    pub nearend_average_blocks: usize,
    pub normal_tuning: SuppressorTuning,
    pub nearend_tuning: SuppressorTuning,
    pub lf_smoothing_during_initial_phase: bool,
    pub last_permanent_lf_smoothing_band: usize,
    pub last_lf_smoothing_band: usize,
    pub last_lf_band: usize,
    pub first_hf_band: usize,
    pub dominant_nearend_detection: DominantNearendDetectionConfig,
    pub high_bands_suppression: HighBandsSuppressionConfig,
    pub high_frequency_suppression: HighFrequencySuppressionConfig,
    pub floor_first_increase: f32,
    pub conservative_hf_suppression: bool,
    /// 实验开关（上游无此字段）：emr 掩蔽用 max(N2, α·平滑近端谱)——
    /// 让近端语音参与掩蔽，双讲时减少抑制（调研报告 L4 杠杆）。
    pub nearend_masker: bool,
    /// 近端掩蔽强度系数 α（0~1]：越小 → 残留回声越少、近端保真略降。
    pub nearend_masker_alpha: f32,
}

impl Default for SuppressorConfig {
    fn default() -> Self {
        Self {
            nearend_average_blocks: 4,
            normal_tuning: SuppressorTuning {
                lf: MaskingThresholds {
                    enr_transparent: 0.3,
                    enr_suppress: 0.4,
                    emr_transparent: 0.3,
                },
                hf: MaskingThresholds {
                    enr_transparent: 0.07,
                    enr_suppress: 0.1,
                    emr_transparent: 0.3,
                },
                max_inc_factor: 2.0,
                max_dec_factor_lf: 0.25,
            },
            nearend_tuning: SuppressorTuning {
                lf: MaskingThresholds {
                    enr_transparent: 1.09,
                    enr_suppress: 1.1,
                    emr_transparent: 0.3,
                },
                hf: MaskingThresholds {
                    enr_transparent: 0.1,
                    enr_suppress: 0.3,
                    emr_transparent: 0.3,
                },
                max_inc_factor: 2.0,
                max_dec_factor_lf: 0.25,
            },
            lf_smoothing_during_initial_phase: true,
            last_permanent_lf_smoothing_band: 0,
            last_lf_smoothing_band: 5,
            last_lf_band: 5,
            first_hf_band: 8,
            dominant_nearend_detection: DominantNearendDetectionConfig {
                enr_threshold: 0.25,
                enr_exit_threshold: 10.0,
                snr_threshold: 30.0,
                hold_duration: 50,
                trigger_threshold: 12,
                use_during_initial_phase: true,
                use_unbounded_echo_spectrum: true,
            },
            high_bands_suppression: HighBandsSuppressionConfig {
                enr_threshold: 1.0,
                max_gain_during_echo: 1.0,
                anti_howling_activation_threshold: 400.0,
                anti_howling_gain: 1.0,
            },
            high_frequency_suppression: HighFrequencySuppressionConfig {
                limiting_gain_band: 16,
                bands_in_limiting_gain: 1,
            },
            floor_first_increase: 1e-5,
            conservative_hf_suppression: false,
            nearend_masker: false,
            nearend_masker_alpha: 1.0,
        }
    }
}

// —— echo_model（residual_echo_estimator.cc）——

/// 渲染噪声门功率（`noise_gate_power`）。
pub const ECHO_MODEL_NOISE_GATE_POWER: f32 = 27509.42;
/// 渲染噪声门斜率（`noise_gate_slope`）。
pub const ECHO_MODEL_NOISE_GATE_SLOPE: f32 = 0.3;
/// 平稳噪声门斜率（`stationary_gate_slope`）。
pub const ECHO_MODEL_STATIONARY_GATE_SLOPE: f32 = 10.0;
/// 渲染噪声底下限（`min_noise_floor_power`，X2_noise_floor 初值）。
pub const ECHO_MODEL_MIN_NOISE_FLOOR_POWER: f32 = 1638400.0;
/// X2_noise_floor 保持块数（`noise_floor_hold`）。
pub const ECHO_MODEL_NOISE_FLOOR_HOLD: usize = 50;
/// 回声生成功率窗口（render_pre/post_window_size，块）。
pub const ECHO_MODEL_RENDER_PRE_WINDOW_SIZE: usize = 1;
pub const ECHO_MODEL_RENDER_POST_WINDOW_SIZE: usize = 1;
/// 非线性模式是否建模混响（`model_reverb_in_nonlinear_mode`）。
pub const ECHO_MODEL_MODEL_REVERB_IN_NONLINEAR_MODE: bool = true;
/// X2_noise_floor 上漂因子（residual_echo_estimator.cc:355）。
pub const ECHO_MODEL_NOISE_FLOOR_LEAK: f32 = 1.1;

// —— ep_strength ——
/// 默认回声路径增益（`ep_strength.default_gain`，幅度域）。
pub const EP_STRENGTH_DEFAULT_GAIN: f32 = 1.0;
/// 默认混响衰减（`ep_strength.default_len`，≥0 → 固定不自适应）。
pub const EP_STRENGTH_DEFAULT_LEN: f32 = 0.83;
/// 透明模式回声路径增益（`kDefaultTransparentModeGain`，幅度域）。
pub const TRANSPARENT_MODE_GAIN: f32 = 0.01;

// —— ERLE（简版估计器）——
/// ERLE 下限。
pub const ERLE_MIN: f32 = 1.0;
/// ERLE 上限：低半带（bins 0..32）。
pub const ERLE_MAX_L: f32 = 4.0;
/// ERLE 上限：高半带（bins 32..65）。
pub const ERLE_MAX_H: f32 = 1.5;
/// 无界 ERLE 上限（`kUnboundedErleMax`）。
pub const ERLE_UNBOUNDED_MAX: f32 = 1e5;

// —— echo_audibility（suppression_gain.cc GetMinGain/WeightEchoForAudibility）——
/// 过减目标功率：低噪声 render。
pub const ECHO_AUDIBILITY_LOW_RENDER_LIMIT: f32 = 4.0 * 64.0; // 256
/// 过减目标功率：正常 render。
pub const ECHO_AUDIBILITY_NORMAL_RENDER_LIMIT: f32 = 64.0;
/// 可闻性地板功率（`floor_power` = 2·64）。
pub const ECHO_AUDIBILITY_FLOOR_POWER: f32 = 2.0 * 64.0; // 128
/// 可闻性阈值倍率（lf/mf/hf 同值 10）。
pub const ECHO_AUDIBILITY_THRESHOLD: f32 = 10.0;

// —— 舒适噪声（comfort_noise_generator.cc）——
/// 噪声底 dBFS（`comfort_noise.noise_floor_dbfs`）。
pub const COMFORT_NOISE_FLOOR_DBFS: f32 = -96.03406;
/// N2 初值。
pub const CNG_N2_INITIAL: f32 = 1.0e6;
/// Y2 平滑系数。
pub const CNG_Y2_SMOOTHING: f32 = 0.1;
/// N2 min 更新系数（0.9 新 + 0.1 旧）。
pub const CNG_N2_MIN_UPDATE: f32 = 0.9;
/// N2 上漂因子。
pub const CNG_N2_UPWARD_DRIFT: f32 = 1.0002;
/// N2 更新启动门槛（块）。
pub const CNG_N2_COUNTER_THRESHOLD: i32 = 50;
/// initial 期时长（块）。
pub const CNG_N2_INITIAL_BLOCKS: i32 = 1000;
/// initial 期跟踪系数。
pub const CNG_N2_INITIAL_ALPHA: f32 = 0.001;
/// LCG 种子与乘子（GenerateRandomSinTableIndices）。
pub const CNG_SEED: u32 = 42;
pub const CNG_LCG_MULTIPLIER: u32 = 69069;

// —— 抑制器（suppression_gain.cc / suppression_filter.cc）——
/// LowNoiseRender 判定阈值（50²·64）。
pub const LOW_NOISE_RENDER_THRESHOLD: f32 = 50.0 * 50.0 * 64.0; // 160000
/// LowNoiseRender 峰值/均值比阈值。
pub const LOW_NOISE_RENDER_PEAK_FACTOR: f32 = 3.0;
/// 输出钳位。
pub const SUPPRESSION_CLAMP: f32 = 32768.0;
/// 高带噪声缩放系数。
pub const HIGH_BAND_NOISE_SCALING: f32 = 0.4;
