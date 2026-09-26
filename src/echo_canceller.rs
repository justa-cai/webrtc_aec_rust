//! 顶层帧 API，对照 `modules/audio_processing/aec3/echo_canceller3.{h,cc}`。
//!
//! 10 ms（160 样本）帧进出；内部样本 FIFO 组块（160 = 2.5×64，每 2 帧 5 块，
//! 与上游 FrameBlocker/BlockFramer 语义等价）。render 帧先入队，
//! 处理 capture 帧前先排空（保持上游 EmptyRenderQueue 的顺序）。
//!
//! 帧级饱和检测：任一样本 |y| ≥ 32700（`DetectSaturation`）。

use crate::block_processor::{BlockProcessor, RenderEvent};
use crate::constants::{BLOCK_SIZE, CAPTURE_SATURATION_LIMIT, FRAME_SIZE};

/// 每 10 ms capture 帧处理后返回的诊断信息（聚合该帧内所有块）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameMetrics {
    /// 本帧 |y| ≥ 32700。
    pub capture_saturation: bool,
    /// render 缓冲溢出（触发过整体复位）。
    pub render_overrun: bool,
    /// render 缓冲下溢。
    pub render_underrun: bool,
    /// 缓冲复位事件（kBufferFlush）。
    pub buffer_flush: bool,
    /// 检测到新时延（kNewDetectedDelay）。
    pub delay_change: bool,
    /// 当前延迟估计（块）。
    pub delay_blocks: Option<usize>,
    /// 线性估计是否可用（AecState 门控）。
    pub usable_linear_estimate: bool,
    /// 初始状态（2.5 s）是否仍在。
    pub initial_state_active: bool,
    /// 近端主导检测状态（NLP 层）。
    pub nearend_state: bool,
}

/// 16 kHz 单声道线性 AEC（AEC3 线性部分全保真移植）。
pub struct EchoCanceller {
    /// 未处理的 capture 输入。
    pending: Vec<f32>,
    /// 已处理的输出。
    ready: Vec<f32>,
    /// 已处理的线性输出。
    linear_ready: Vec<f32>,
    /// render 余数（不足一块）。
    render_remainder: Vec<f32>,
    block_processor: BlockProcessor,
    /// 帧级饱和检测标志（对下一块的处理生效，与上游 AnalyzeCapture 时序一致）。
    saturated_microphone_signal: bool,
    // ERLE 统计（演示/评估用）
    sum_y2: f64,
    sum_e2: f64,
    erle_ready: bool,
}

impl Default for EchoCanceller {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoCanceller {
    pub fn new() -> Self {
        Self {
            pending: Vec::with_capacity(FRAME_SIZE + BLOCK_SIZE),
            ready: Vec::with_capacity(FRAME_SIZE + BLOCK_SIZE),
            linear_ready: Vec::with_capacity(FRAME_SIZE + BLOCK_SIZE),
            render_remainder: Vec::with_capacity(BLOCK_SIZE),
            block_processor: BlockProcessor::new(),
            saturated_microphone_signal: false,
            sum_y2: 0.0,
            sum_e2: 0.0,
            erle_ready: false,
        }
    }

    /// 送入 10 ms 远端（render）帧。须在对应 capture 帧之前调用。
    pub fn push_render_frame(&mut self, render_frame: &[f32; FRAME_SIZE]) -> RenderEvent {
        let mut stream = self.render_remainder.clone();
        stream.extend_from_slice(render_frame);
        let mut event = RenderEvent::None;
        let mut offset = 0usize;
        while offset + BLOCK_SIZE <= stream.len() {
            let block: [f32; BLOCK_SIZE] =
                stream[offset..offset + BLOCK_SIZE].try_into().unwrap();
            let e = self.block_processor.buffer_render(&block);
            if e == RenderEvent::Overrun {
                event = RenderEvent::Overrun;
            }
            offset += BLOCK_SIZE;
        }
        self.render_remainder = stream[offset..].to_vec();
        event
    }

    /// 处理 10 ms 近端（capture）帧，原地替换为线性 AEC 输出 e。
    pub fn process_capture_frame(
        &mut self,
        capture_frame: &mut [f32; FRAME_SIZE],
    ) -> FrameMetrics {
        self.process_impl(capture_frame, None)
    }

    /// 同上，另外导出线性滤波器输出。
    pub fn process_capture_frame_with_linear_output(
        &mut self,
        capture_frame: &mut [f32; FRAME_SIZE],
        linear_output: &mut [f32; FRAME_SIZE],
    ) -> FrameMetrics {
        self.process_impl(capture_frame, Some(linear_output))
    }

    fn process_impl(
        &mut self,
        capture_frame: &mut [f32; FRAME_SIZE],
        mut linear_output: Option<&mut [f32; FRAME_SIZE]>,
    ) -> FrameMetrics {
        // 帧级饱和检测（上游 DetectSaturation：对后续处理生效）
        let saturated = capture_frame
            .iter()
            .any(|v| *v >= CAPTURE_SATURATION_LIMIT || *v <= -CAPTURE_SATURATION_LIMIT);

        self.pending.extend_from_slice(capture_frame);
        let want_linear = linear_output.is_some();

        let mut metrics = FrameMetrics {
            capture_saturation: saturated,
            ..Default::default()
        };
        let mut last_nearend_state = false;

        while self.pending.len() >= BLOCK_SIZE {
            let mut block: [f32; BLOCK_SIZE] = self.pending[..BLOCK_SIZE].try_into().unwrap();
            let y2: f64 = block.iter().map(|v| (*v as f64) * (*v as f64)).sum();

            let mut linear_block = if want_linear {
                Some([0.0f32; BLOCK_SIZE])
            } else {
                None
            };
            let result = self.block_processor.process_capture(
                false, // API 级增益变化标志（上游由外部传入，本实现恒 false）
                self.saturated_microphone_signal,
                linear_block.as_mut(),
                &mut block,
            );

            let e2: f64 = block.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            self.sum_y2 += y2;
            self.sum_e2 += e2;
            self.erle_ready = true;

            if result.delay_change {
                metrics.delay_change = true;
            }
            self.pending.drain(..BLOCK_SIZE);
            last_nearend_state = result.nearend_state;
            self.ready.extend_from_slice(&block);
            if let Some(lb) = linear_block {
                self.linear_ready.extend_from_slice(&lb);
            }
        }
        self.saturated_microphone_signal = saturated;

        metrics.usable_linear_estimate = self
            .block_processor
            .echo_remover()
            .aec_state()
            .usable_linear_estimate();
        metrics.nearend_state = last_nearend_state;
        metrics.initial_state_active = self
            .block_processor
            .echo_remover()
            .aec_state()
            .initial_state_active();
        metrics.delay_blocks = self.block_processor.delay_blocks();

        // 从已处理队列取回输出；不足一帧的部分留待下一帧
        //（首个输出帧因此有最多 32 样本的固定时延）。
        let take = self.ready.len().min(FRAME_SIZE);
        capture_frame[..take].copy_from_slice(&self.ready[..take]);
        for v in capture_frame[take..].iter_mut() {
            *v = 0.0;
        }
        self.ready.drain(..take);
        if let Some(lo) = linear_output.as_deref_mut() {
            lo[..take].copy_from_slice(&self.linear_ready[..take]);
            for v in lo[take..].iter_mut() {
                *v = 0.0;
            }
            self.linear_ready.drain(..take);
        }

        metrics
    }

    /// 全带 ERLE（10·log10(Σy²/Σe²)），评估用。
    pub fn erle_db(&self) -> Option<f32> {
        if !self.erle_ready || self.sum_e2 <= 0.0 {
            return None;
        }
        Some((10.0 * (self.sum_y2 / self.sum_e2).log10()) as f32)
    }

    /// 分段 ERLE：重置累计器。
    pub fn reset_erle_stats(&mut self) {
        self.sum_y2 = 0.0;
        self.sum_e2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T23（调整）：帧 API 排空守恒——送入 N 帧后逐帧取出，
    /// 处理输出的总样本数与输入一致（至多多一帧的流水时延）、全部有限。
    /// （帧路径按帧批量插入 render，与逐块交错是不同时序，逐位相等不成立。）
    #[test]
    fn frame_api_drains_correctly() {
        let mut aec = EchoCanceller::new();
        let mut state = 0x1234u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) as f32 / (1u64 << 30) as f32 - 1.0) * 2000.0
        };
        let mut total_out = 0usize;
        let mut nonzero_tail = 0usize;
        let n_frames = 20usize;
        for f in 0..n_frames {
            let mut rf = [0.0f32; FRAME_SIZE];
            let mut cf = [0.0f32; FRAME_SIZE];
            for k in 0..FRAME_SIZE {
                let v = next();
                rf[k] = v;
                cf[k] = v * 0.5; // 简单相关信号
            }
            aec.push_render_frame(&rf);
            aec.process_capture_frame(&mut cf);
            let real = cf.iter().filter(|v| **v != 0.0).count();
            if f > 0 {
                // 除首帧（32 样本流水时延）外应整帧有输出
                assert!(real >= FRAME_SIZE - BLOCK_SIZE, "帧 {} 仅 {} 样本", f, real);
            }
            nonzero_tail += real;
            total_out += cf.len();
            assert!(cf.iter().all(|v| v.is_finite()));
        }
        assert_eq!(total_out, n_frames * FRAME_SIZE);
        // 流水时延内的零样本不超过一帧
        assert!(n_frames * FRAME_SIZE - nonzero_tail <= FRAME_SIZE);
    }
}
