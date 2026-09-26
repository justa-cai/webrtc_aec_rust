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
