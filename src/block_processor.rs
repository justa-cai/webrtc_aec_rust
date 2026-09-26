//! 块处理器，对照 `modules/audio_processing/aec3/block_processor.{h,cc}`。
//!
//! 每块 capture：先排空新到的 render 块 → 时延估计与对齐 → EchoRemover。
//! 延迟变化/缓冲事件 → `EchoPathVariability` → 滤波器复位链。
//!
//! 未移植：外部时延估计路径（`use_external_delay_estimator`）、指标上报。

use crate::constants::BLOCK_SIZE;
use crate::delay_estimate::DelayEstimate;
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::echo_remover::{BlockResult, EchoRemover};
use crate::render_delay_buffer::{BufferingEvent, RenderDelayBuffer};
use crate::render_delay_controller::RenderDelayController;

/// render 帧处理结果（`RenderDelayBuffer::BufferingEvent` 的公共别名）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderEvent {
    None,
    Overrun,
}

/// 块处理器（`BlockProcessorImpl`）。单声道。
pub struct BlockProcessor {
    render_buffer: RenderDelayBuffer,
    delay_controller: RenderDelayController,
    echo_remover: EchoRemover,
    capture_properly_started: bool,
    render_properly_started: bool,
    pending_render_event: BufferingEvent,
    estimated_delay: Option<DelayEstimate>,
}

impl BlockProcessor {
    pub fn new() -> Self {
        Self::with_suppressor_config(Default::default())
    }

    /// 注入抑制器配置（双讲调优实验用）。
    pub fn with_suppressor_config(suppressor_config: crate::constants::SuppressorConfig) -> Self {
        Self {
            render_buffer: RenderDelayBuffer::new(),
            delay_controller: RenderDelayController::new(),
            echo_remover: EchoRemover::with_suppressor_config(suppressor_config),
            capture_properly_started: false,
            render_properly_started: false,
            pending_render_event: BufferingEvent::None,
            estimated_delay: None,
        }
    }

    /// 缓冲一块 render（`BufferRender`）。
    pub fn buffer_render(&mut self, block: &[f32; BLOCK_SIZE]) -> RenderEvent {
        let event = self.render_buffer.insert(block);
        self.render_properly_started = true;
        match event {
            BufferingEvent::RenderOverrun => {
                self.pending_render_event = BufferingEvent::RenderOverrun;
                RenderEvent::Overrun
            }
            _ => RenderEvent::None,
        }
    }

    /// 处理一块 capture（`ProcessCapture`）。`capture` 原地替换为输出。
    pub fn process_capture(
        &mut self,
        echo_path_gain_change: bool,
        capture_signal_saturation: bool,
        linear_output: Option<&mut [f32; BLOCK_SIZE]>,
        capture: &mut [f32; BLOCK_SIZE],
    ) -> BlockResult {
        if !self.render_properly_started {
            // 尚无 render：跳过 capture 处理
            self.render_buffer.handle_skipped_capture_processing();
            return BlockResult::default();
        }
        if !self.capture_properly_started {
            self.capture_properly_started = true;
            self.render_buffer.reset();
            self.delay_controller.reset(true);
        }

        let mut variability = EchoPathVariability::new(
            echo_path_gain_change,
            DelayAdjustment::None,
            false,
        );

        // 上一次 Insert 的 overrun → 缓冲整体复位已发生，视为路径变化
        if self.pending_render_event == BufferingEvent::RenderOverrun {
            variability.delay_change = DelayAdjustment::BufferFlush;
            self.delay_controller.reset(true);
        }
        self.pending_render_event = BufferingEvent::None;

        // 推进读索引
        let buffer_event = self.render_buffer.prepare_capture_processing();
        if buffer_event == BufferingEvent::RenderUnderrun {
            self.delay_controller.reset(false);
        }

        // 时延估计与对齐
        self.estimated_delay = self.delay_controller.get_delay(
            self.render_buffer.get_downsampled_render_buffer(),
            capture,
        );
        if let Some(d) = self.estimated_delay {
            if self.render_buffer.align_from_delay(d.delay) {
                variability.delay_change = DelayAdjustment::NewDetectedDelay;
            }
        }
        variability.clock_drift = self.delay_controller.has_clockdrift();

        // 线性回声消除
        self.echo_remover.process_capture(
            &variability,
            capture_signal_saturation,
            self.estimated_delay,
            &self.render_buffer.get_render_buffer(),
            linear_output,
            capture,
        )
    }

    /// 当前延迟估计（块）。
    pub fn delay_blocks(&self) -> Option<usize> {
        self.estimated_delay.map(|d| d.delay)
    }

    pub fn echo_remover(&self) -> &EchoRemover {
        &self.echo_remover
    }
}

impl Default for BlockProcessor {
    fn default() -> Self {
        Self::new()
    }
}
