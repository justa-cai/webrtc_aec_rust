//! AEC 状态（lite 版），对照 `modules/audio_processing/aec3/aec_state.{h,cc}` 与
//! `subtractor_output_analyzer.{h,cc}`。
//!
//! 只保留线性路径 gating 所需：
//! - `SubtractorOutputAnalyzer`：从能量比判定滤波器收敛/发散；
//! - `InitialState`：2.5 s 强非饱和 render 后退出初始状态（触发滤波器加长）；
//! - `FilterDelay`：直达路径滤波器延迟（前 2 s 且有外部时延时强制为 headroom=0）；
//! - `FilteringQualityAnalyzer`：`UsableLinearEstimate` 的门控。
//!
//! 未移植（上游仅用于抑制器/指标）：ERLE/ERL 估计器、饱和回声检测、混响模型、
//! 透明模式（`transparent_mode` 恒为 false）。

use crate::constants::{
    CONVERGENCE_THRESHOLD, CONVERGENCE_THRESHOLD_LOW_LEVEL, DELAY_HEADROOM_SAMPLES,
    DIVERGENCE_THRESHOLD, FFT_LENGTH_BY_2_PLUS_1, INITIAL_STATE_BLOCKS, NUM_BLOCKS_PER_SECOND,
    BLOCK_SIZE,
};
use crate::delay_estimate::DelayEstimate;
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::filter_analyzer::FilterAnalyzer;
use crate::render_delay_buffer::RenderBufferView;
use crate::subtractor_output::SubtractorOutput;

/// 收敛/发散判定（`SubtractorOutputAnalyzer`）。
#[derive(Clone, Debug, Default)]
pub struct SubtractorOutputAnalyzer {
    filters_converged: bool,
    any_coarse_filter_converged: bool,
    filter_diverged: bool,
}

impl SubtractorOutputAnalyzer {
    pub fn update(&mut self, output: &SubtractorOutput) {
        let refined_converged = output.e2_refined < 0.5 * output.y2
            && output.y2 > CONVERGENCE_THRESHOLD;
        let coarse_converged_strict = output.e2_coarse < 0.05 * output.y2
            && output.y2 > CONVERGENCE_THRESHOLD;
        let coarse_converged_relaxed = output.e2_coarse < 0.3 * output.y2
            && output.y2 > CONVERGENCE_THRESHOLD_LOW_LEVEL;
        self.filter_diverged = output.e2_refined.min(output.e2_coarse) > 1.5 * output.y2
            && output.y2 > DIVERGENCE_THRESHOLD;
        self.filters_converged = refined_converged || coarse_converged_strict;
        self.any_coarse_filter_converged = coarse_converged_strict || coarse_converged_relaxed;
    }

    pub fn handle_echo_path_change(&mut self) {
        self.filters_converged = false;
    }

    pub fn converged_filters(&self) -> bool {
        self.filters_converged
    }

    pub fn any_coarse_filter_converged(&self) -> bool {
        self.any_coarse_filter_converged
    }

    pub fn diverged_filters(&self) -> bool {
        self.filter_diverged
    }
}

/// 初始状态（`AecState::InitialState`）。
#[derive(Clone, Debug)]
struct InitialState {
    transition_triggered: bool,
    initial_state: bool,
    strong_not_saturated_render_blocks: usize,
}

impl InitialState {
    fn new() -> Self {
        Self {
            transition_triggered: false,
            initial_state: true,
            strong_not_saturated_render_blocks: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn update(&mut self, active_render: bool, saturated_capture: bool) {
        if active_render && !saturated_capture {
            self.strong_not_saturated_render_blocks += 1;
        }
        let prev = self.initial_state;
        // conservative_initial_phase 默认 false；initial_state_seconds = 2.5
        self.initial_state =
            self.strong_not_saturated_render_blocks < INITIAL_STATE_BLOCKS;
        self.transition_triggered = !self.initial_state && prev;
    }

    fn initial_state_active(&self) -> bool {
        self.initial_state
    }

    fn transition_triggered(&self) -> bool {
        self.transition_triggered
    }
}

/// 直达路径延迟（`AecState::FilterDelay`）。
#[derive(Clone, Debug)]
struct FilterDelayState {
    /// delay_headroom_samples / 64 = 0 块（整除）。
    delay_headroom_blocks: i32,
    filter_delay_blocks: i32,
    external_delay: Option<DelayEstimate>,
}

impl FilterDelayState {
    fn new() -> Self {
        Self {
            delay_headroom_blocks: (DELAY_HEADROOM_SAMPLES / BLOCK_SIZE) as i32,
            filter_delay_blocks: (DELAY_HEADROOM_SAMPLES / BLOCK_SIZE) as i32,
            external_delay: None,
        }
    }

    fn reset(&mut self) {
        self.filter_delay_blocks = self.delay_headroom_blocks;
    }

    fn update(
        &mut self,
        analyzer_filter_delay_blocks: i32,
        external_delay: Option<DelayEstimate>,
        blocks_with_proper_filter_adaptation: usize,
    ) {
        if let Some(d) = external_delay {
            self.external_delay = Some(d);
        }
        // 滤波器可能尚未收敛时（前 2 s），若有外部时延则强制用 headroom 猜测。
        let may_not_have_converged =
            blocks_with_proper_filter_adaptation < 2 * NUM_BLOCKS_PER_SECOND;
        if may_not_have_converged && self.external_delay.is_some() {
            self.filter_delay_blocks = self.delay_headroom_blocks;
        } else {
            self.filter_delay_blocks = analyzer_filter_delay_blocks;
        }
    }

    fn min_direct_path_filter_delay(&self) -> i32 {
        self.filter_delay_blocks
    }
}

/// 线性估计可用性门控（`AecState::FilteringQualityAnalyzer`）。
#[derive(Clone, Debug)]
struct FilteringQualityAnalyzer {
    overall_usable_linear_estimates: bool,
    filter_update_blocks_since_reset: usize,
    filter_update_blocks_since_start: usize,
    convergence_seen: bool,
}

impl FilteringQualityAnalyzer {
    fn new() -> Self {
        Self {
            overall_usable_linear_estimates: false,
            filter_update_blocks_since_reset: 0,
            filter_update_blocks_since_start: 0,
            convergence_seen: false,
        }
    }

    fn reset(&mut self) {
        // 只复位 reset 计数器与输出标志；since_start 与 convergence_seen 保留
        //（对照 FilteringQualityAnalyzer::Reset，aec_state.cc:396）。
        self.overall_usable_linear_estimates = false;
        self.filter_update_blocks_since_reset = 0;
    }

    fn update(
        &mut self,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Option<DelayEstimate>,
        any_filter_converged: bool,
    ) {
        let filter_update = active_render && !saturated_capture;
        if filter_update {
            self.filter_update_blocks_since_reset += 1;
            self.filter_update_blocks_since_start += 1;
        }
        self.convergence_seen = self.convergence_seen || any_filter_converged;

        // 启动 0.4 s、复位后 0.2 s 的有效更新块
        let sufficient_at_startup =
            self.filter_update_blocks_since_start > (NUM_BLOCKS_PER_SECOND / 5 * 2) as usize;
        let sufficient_at_reset = sufficient_at_startup
            && self.filter_update_blocks_since_reset > (NUM_BLOCKS_PER_SECOND / 5) as usize;

        self.overall_usable_linear_estimates = sufficient_at_startup && sufficient_at_reset;
        // 外部时延或见过收敛
        self.overall_usable_linear_estimates =
            self.overall_usable_linear_estimates && (external_delay.is_some() || self.convergence_seen);
        // 透明模式（本移植恒 false）
        self.overall_usable_linear_estimates =
            self.overall_usable_linear_estimates && !transparent_mode;
    }

    fn linear_filter_usable(&self) -> bool {
        self.overall_usable_linear_estimates
    }
}

/// AEC 状态（`AecState`，lite）。单声道。
pub struct AecState {
    use_linear_filter: bool,
    subtractor_output_analyzer: SubtractorOutputAnalyzer,
    filter_analyzer: FilterAnalyzer,
    delay_state: FilterDelayState,
    initial_state: InitialState,
    filter_quality_state: FilteringQualityAnalyzer,
    saturated_capture: bool,
    /// 上块活跃 render（供指标）。
    active_render: bool,
}

impl AecState {
    pub fn new() -> Self {
        Self {
            use_linear_filter: true, // config.filter.use_linear_filter
            subtractor_output_analyzer: SubtractorOutputAnalyzer::default(),
            filter_analyzer: FilterAnalyzer::new(),
            delay_state: FilterDelayState::new(),
            initial_state: InitialState::new(),
            filter_quality_state: FilteringQualityAnalyzer::new(),
            saturated_capture: false,
            active_render: false,
        }
    }

    pub fn update_capture_saturation(&mut self, saturated: bool) {
        self.saturated_capture = saturated;
    }

    pub fn saturated_capture(&self) -> bool {
        self.saturated_capture
    }

    /// 线性估计是否可用（`UsableLinearEstimate`；本版本与
    /// `UseLinearFilterOutput` 同式）。
    pub fn usable_linear_estimate(&self) -> bool {
        self.filter_quality_state.linear_filter_usable() && self.use_linear_filter
    }

    /// 直达路径滤波器延迟（块，`MinDirectPathFilterDelay`）。
    pub fn min_direct_path_filter_delay(&self) -> i32 {
        self.delay_state.min_direct_path_filter_delay()
    }

    pub fn converged_filters(&self) -> bool {
        self.subtractor_output_analyzer.converged_filters()
    }

    pub fn initial_state_active(&self) -> bool {
        self.initial_state.initial_state_active()
    }

    /// 调试：活跃非饱和 render 块计数（初始状态退出阈值 625）。
    pub fn debug_strong_blocks(&self) -> usize {
        self.initial_state.debug_strong_blocks()
    }

    /// 初始状态→正常状态的切换沿（`TransitionTriggered`）。
    pub fn transition_triggered(&self) -> bool {
        self.initial_state.transition_triggered()
    }

    /// 回声路径变化（`HandleEchoPathVariability`）：延迟变化全复位。
    pub fn handle_echo_path_change(&mut self, variability: &EchoPathVariability) {
        if variability.delay_change != DelayAdjustment::None {
            self.filter_analyzer.reset();
            self.saturated_capture = false;
            self.initial_state.reset();
            self.filter_quality_state.reset();
            self.delay_state.reset();
            // 未移植：ERLE/ERL 复位、透明模式复位
        }
        self.subtractor_output_analyzer.handle_echo_path_change();
    }

    /// 每块更新（`AecState::Update`，保留 gating 相关部分）。
    pub fn update(
        &mut self,
        external_delay: Option<DelayEstimate>,
        filter_impulse_responses: &[f32],
        render_buffer: &RenderBufferView,
        e2_refined: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        subtractor_output: &SubtractorOutput,
    ) {
        // 1. 收敛/发散
        self.subtractor_output_analyzer.update(subtractor_output);
        let any_filter_converged = self.subtractor_output_analyzer.converged_filters();

        // 2. 滤波器分析（峰位/一致性/增益）
        self.filter_analyzer.update(filter_impulse_responses, render_buffer);

        // 3. 延迟状态
        self.delay_state.update(
            self.filter_analyzer.min_filter_delay_blocks(),
            external_delay,
            self.initial_state.strong_not_saturated_render_blocks_public(),
        );

        // 4. 活跃 render 计数（对齐块能量）
        let aligned_block =
            render_buffer.get_block(-(self.min_direct_path_filter_delay() as isize));
        let energy: f32 = aligned_block.iter().map(|v| v * v).sum();
        let active_render = energy > crate::constants::ACTIVE_RENDER_BLOCK_ENERGY;
        self.active_render = active_render;

        // 5. 初始状态
        self.initial_state.update(active_render, self.saturated_capture);

        // 6. 线性估计门控（透明模式恒 false）
        self.filter_quality_state.update(
            active_render,
            false,
            self.saturated_capture,
            external_delay,
            any_filter_converged,
        );
        // 未移植：ERLE/ERL 估计（e2_refined/y2 仅上游指标消费）、饱和回声检测、
        // 混响模型
        let _ = (e2_refined, y2);
    }
}

impl InitialState {
    fn strong_not_saturated_render_blocks_public(&self) -> usize {
        self.strong_not_saturated_render_blocks
    }
    /// 调试：活跃非饱和块计数。
    pub fn debug_strong_blocks(&self) -> usize {
        self.strong_not_saturated_render_blocks
    }
}

impl Default for AecState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T20（单元部分）：门控时序。
    #[test]
    fn usable_linear_estimate_timing() {
        let mut q = FilteringQualityAnalyzer::new();
        let ext = Some(DelayEstimate::new(
            crate::delay_estimate::DelayQuality::Refined,
            25,
        ));
        // 启动 100 块内 false
        for i in 0..100 {
            q.update(true, false, false, ext, false);
            assert!(!q.linear_filter_usable(), "i={}", i);
        }
        q.update(true, false, false, ext, false);
        assert!(q.linear_filter_usable());

        // 复位后 50 块恢复
        q.reset();
        for i in 0..50 {
            q.update(true, false, false, ext, false);
            assert!(!q.linear_filter_usable(), "reset i={}", i);
        }
        q.update(true, false, false, ext, false);
        assert!(q.linear_filter_usable());
    }

    /// 无外部时延且从未收敛 → 不可用。
    #[test]
    fn no_delay_no_convergence_gated() {
        let mut q = FilteringQualityAnalyzer::new();
        for _ in 0..200 {
            q.update(true, false, false, None, false);
        }
        assert!(!q.linear_filter_usable());
        // 见过一次收敛即可放行
        q.update(true, false, false, None, true);
        assert!(q.linear_filter_usable());
    }

    /// 收敛判定常量。
    #[test]
    fn convergence_flags() {
        let mut a = SubtractorOutputAnalyzer::default();
        let mut o = SubtractorOutput::new();
        o.y2 = 1.0e6;
        o.e2_refined = 0.1 * o.y2; // < 0.5·y2 → refined 收敛
        a.update(&o);
        assert!(a.converged_filters());
        assert!(!a.diverged_filters());

        o.e2_refined = 2.0 * o.y2; // min(e2) > 1.5·y2 → 发散
        o.e2_coarse = 2.5 * o.y2;
        a.update(&o);
        assert!(a.diverged_filters());
        assert!(!a.converged_filters());
    }
}
