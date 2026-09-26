//! render 信号的多级缓冲与延迟对齐，对照
//! `modules/audio_processing/aec3/render_delay_buffer.{h,cc}`、`render_buffer.{h,cc}`。
//!
//! # 结构
//!
//! - `blocks`/`spectra`/`ffts` 三条同步推进的环形缓冲（长度 167 块）：
//!   - blocks 的 write **正向**步进；spectra/ffts 的 write **负向**步进；
//!   - `ApplyTotalDelay` 时 blocks.read = write − delay，spectra/ffts.read = write + delay
//!     （方向相反恰好指向同一逻辑块）。
//! - `low_rate`：降采样（×4）render 环形缓冲（2448 样本），每块以 −16 步进写入，
//!   且 16 个样本**逆序**存放（供匹配滤波器使用）。
//!
//! # 未移植
//!
//! - `render_activity_`（上游仅供残余回声估计/可听性判断）
//! - API 抖动跟踪（仅打日志）
//! - 外部音频缓冲延迟路径（`SetAudioBufferDelay`/`AlignFromExternalDelay`，
//!   对应 `use_external_delay_estimator` 配置，本实现不用）
//! - `render_power_gain_db`（默认 0 dB，增益恒 1）

use crate::constants::{
    BLOCK_SIZE, DELAY_DEFAULT_BLOCKS, DELAY_DOWN_SAMPLING_FACTOR, DELAY_SUB_BLOCK_SIZE,
    DOWNSAMPLED_BUFFER_SIZE, EXCESS_RENDER_DETECTION_INTERVAL_BLOCKS,
    FFT_LENGTH_BY_2_PLUS_1, FILTER_REFINED_LENGTH_BLOCKS, MAX_ALLOWED_EXCESS_RENDER_BLOCKS,
    RENDER_DELAY_BUFFER_SIZE,
};
use crate::decimator::Decimator;
use crate::fft::Aec3Fft;
use crate::fft_data::FftData;
use crate::ring::{DownsampledRenderBuffer, Ring};

/// 缓冲事件（`RenderDelayBuffer::BufferingEvent`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferingEvent {
    None,
    /// render 数据不足（capture 来时没有新 render 块）。
    RenderUnderrun,
    /// render 数据过量（插入多于消费）→ 触发整体复位。
    RenderOverrun,
}

/// render 延迟缓冲（`RenderDelayBufferImpl`）。单声道、16 kHz。
pub struct RenderDelayBuffer {
    blocks: Ring<[f32; BLOCK_SIZE]>,
    spectra: Ring<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    ffts: Ring<FftData>,
    /// `AlignFromDelay` 设置的延迟（块）；None = 尚未对齐（初始态）。
    delay: Option<usize>,
    low_rate: DownsampledRenderBuffer,
    decimator: Decimator,
    fft: Aec3Fft,
    render_ds: [f32; DELAY_SUB_BLOCK_SIZE],
    /// `MaxDelay() = 块环长 − 1 − headroom`。
    buffer_headroom: usize,
    min_latency_blocks: usize,
    excess_render_detection_counter: usize,
}

impl Default for RenderDelayBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderDelayBuffer {
    pub fn new() -> Self {
        let mut s = Self {
            blocks: Ring::new(RENDER_DELAY_BUFFER_SIZE, [0.0; BLOCK_SIZE]),
            spectra: Ring::new(RENDER_DELAY_BUFFER_SIZE, [0.0; FFT_LENGTH_BY_2_PLUS_1]),
            ffts: Ring::new(RENDER_DELAY_BUFFER_SIZE, FftData::default()),
            delay: Some(DELAY_DEFAULT_BLOCKS),
            low_rate: DownsampledRenderBuffer::new(
                DOWNSAMPLED_BUFFER_SIZE,
                DELAY_SUB_BLOCK_SIZE,
            ),
            decimator: Decimator::new(DELAY_DOWN_SAMPLING_FACTOR),
            fft: Aec3Fft::new(),
            render_ds: [0.0; DELAY_SUB_BLOCK_SIZE],
            buffer_headroom: FILTER_REFINED_LENGTH_BLOCKS,
            min_latency_blocks: 0,
            excess_render_detection_counter: 0,
        };
        s.reset();
        s
    }

    /// 复位缓冲延迟并清除已报告延迟（`RenderDelayBufferImpl::Reset`）。
    pub fn reset(&mut self) {
        self.min_latency_blocks = 0;
        self.excess_render_detection_counter = 0;

        // 低速率环的 read 初始化到 write 前一个子块位置。
        self.low_rate
            .set_read(self.low_rate.offset_index(self.low_rate.write(), DELAY_SUB_BLOCK_SIZE as isize));

        // 无外部延迟：先套默认 total delay，再清除 AlignFromDelay 设置的延迟。
        self.apply_total_delay(DELAY_DEFAULT_BLOCKS as isize);
        self.delay = None;
    }

    /// 插入一块 render 数据（`Insert` + `InsertBlock`）。
    pub fn insert(&mut self, block: &[f32; BLOCK_SIZE]) -> BufferingEvent {
        let previous_write = self.blocks.write();

        // 写索引推进：blocks +1，spectra/ffts −1，低速率环 −16。
        self.increment_write_indices();

        // 允许 overrun，发生时整体复位。
        let event = if self.render_overrun() {
            BufferingEvent::RenderOverrun
        } else {
            BufferingEvent::None
        };

        self.insert_block(block, previous_write);

        if event != BufferingEvent::None {
            self.reset();
        }
        event
    }

    /// 无 capture 处理时的计数（`HandleSkippedCaptureProcessing`；上游仅用于日志）。
    pub fn handle_skipped_capture_processing(&mut self) {}

    /// 为处理下一块 capture 推进读索引（`PrepareCaptureProcessing`）。
    pub fn prepare_capture_processing(&mut self) -> BufferingEvent {
        if self.detect_excess_render_blocks() {
            // render 块远多于 capture 块：时延可能滑出滤波器范围，整体复位。
            self.reset();
            return BufferingEvent::RenderOverrun;
        }
        if self.render_underrun() {
            // underrun：低速率环不动，只推进三环读索引（等效延迟减一）。
            self.increment_read_indices();
            if let Some(d) = self.delay {
                if d > 0 {
                    self.delay = Some(d - 1);
                }
            }
            return BufferingEvent::RenderUnderrun;
        }
        self.low_rate
            .update_read(-(DELAY_SUB_BLOCK_SIZE as isize));
        self.increment_read_indices();
        BufferingEvent::None
    }

    /// 设置延迟；返回是否发生变化（`AlignFromDelay`）。
    pub fn align_from_delay(&mut self, delay: usize) -> bool {
        if self.delay == Some(delay) {
            return false;
        }
        self.delay = Some(delay);

        // total delay = 缓冲 latency + 对齐延迟，并限制到允许范围。
        let mut total_delay = self.map_delay_to_total_delay(delay);
        let max_delay = self.max_delay() as isize;
        total_delay = total_delay.clamp(0, max_delay);
        self.apply_total_delay(total_delay);
        true
    }

    /// 当前延迟（块，`Delay()` = `ComputeDelay()`），不含调用抖动。
    pub fn delay(&self) -> usize {
        let latency_blocks = self.buffer_latency() as isize;
        let internal_delay = if self.spectra.read() >= self.spectra.write() {
            self.spectra.read() - self.spectra.write()
        } else {
            self.spectra.size() as isize + self.spectra.read() - self.spectra.write()
        };
        (internal_delay - latency_blocks) as usize
    }

    pub fn max_delay(&self) -> usize {
        self.blocks.size() - 1 - self.buffer_headroom
    }

    /// 降采样 render 缓冲（喂给匹配滤波器）。
    pub fn get_downsampled_render_buffer(&self) -> &DownsampledRenderBuffer {
        &self.low_rate
    }

    /// 给 EchoRemover/Subtractor 的只读视图（`GetRenderBuffer` → `RenderBuffer`）。
    pub fn get_render_buffer(&self) -> RenderBufferView<'_> {
        RenderBufferView {
            blocks: &self.blocks,
            spectra: &self.spectra,
            ffts: &self.ffts,
        }
    }

    // -----------------------------------------------------------------------
    // 内部
    // -----------------------------------------------------------------------

    /// 缓冲 latency：低速率环中未读的块数（`BufferLatency`）。
    pub(crate) fn buffer_latency(&self) -> usize {
        let l = &self.low_rate;
        let latency_samples =
            (l.size as isize + l.read() - l.write()).rem_euclid(l.size as isize);
        (latency_samples / DELAY_SUB_BLOCK_SIZE as isize) as usize
    }

    /// 外部延迟 → total delay（`MapDelayToTotalDelay`）。
    fn map_delay_to_total_delay(&self, external_delay_blocks: usize) -> isize {
        self.buffer_latency() as isize + external_delay_blocks as isize
    }

    /// 按延迟偏移三条环的读指针（`ApplyTotalDelay`）。
    fn apply_total_delay(&mut self, delay: isize) {
        self.blocks
            .set_read(self.blocks.offset_index(self.blocks.write(), -delay));
        self.spectra
            .set_read(self.spectra.offset_index(self.spectra.write(), delay));
        self.ffts
            .set_read(self.ffts.offset_index(self.ffts.write(), delay));
    }

    /// 写一块数据到三条环（`InsertBlock`）。
    fn insert_block(&mut self, block: &[f32; BLOCK_SIZE], previous_write: isize) {
        let write = self.blocks.write();
        *self.blocks.get_mut(write) = *block;

        // 降采样后逆序写入低速率环（源码：copy(ds.rbegin(), ds.rend(), ...+write)）。
        self.decimator.decimate(block, &mut self.render_ds);
        let lr_write = self.low_rate.write() as usize;
        for k in 0..DELAY_SUB_BLOCK_SIZE {
            self.low_rate.buffer[lr_write + k] = self.render_ds[DELAY_SUB_BLOCK_SIZE - 1 - k];
        }

        // 128 点 PaddedFft（矩形窗，[previous_write 块, 当前块]）+ 功率谱。
        let x_old = *self.blocks.get(previous_write);
        let mut x_fft = FftData::default();
        self.fft
            .padded_fft(block, &x_old, crate::fft::Window::Rectangular, &mut x_fft);
        let f_write = self.ffts.write();
        *self.ffts.get_mut(f_write) = x_fft;
        let s_write = self.spectra.write();
        self.ffts
            .get(f_write)
            .spectrum(self.spectra.get_mut(s_write));
    }

    /// 写索引步进（`IncrementWriteIndices`）。
    fn increment_write_indices(&mut self) {
        self.low_rate
            .update_write(-(DELAY_SUB_BLOCK_SIZE as isize));
        self.blocks.inc_write();
        self.spectra.dec_write();
        self.ffts.dec_write();
    }

    /// 三环读索引步进（`IncrementReadIndices`）。
    fn increment_read_indices(&mut self) {
        if self.blocks.read() != self.blocks.write() {
            self.blocks.inc_read();
            self.spectra.dec_read();
            self.ffts.dec_read();
        }
    }

    /// render 数据过量检测（`DetectExcessRenderBlocks`）。
    fn detect_excess_render_blocks(&mut self) -> bool {
        let latency_blocks = self.buffer_latency();
        self.min_latency_blocks = self.min_latency_blocks.min(latency_blocks);
        self.excess_render_detection_counter += 1;
        if self.excess_render_detection_counter >= EXCESS_RENDER_DETECTION_INTERVAL_BLOCKS {
            let detected = self.min_latency_blocks > MAX_ALLOWED_EXCESS_RENDER_BLOCKS;
            self.excess_render_detection_counter = 0;
            self.min_latency_blocks = latency_blocks;
            return detected;
        }
        false
    }

    fn render_overrun(&self) -> bool {
        self.low_rate.read() == self.low_rate.write()
            || self.blocks.read() == self.blocks.write()
    }

    fn render_underrun(&self) -> bool {
        self.low_rate.read() == self.low_rate.write()
    }
}

/// EchoRemover/Subtractor 对 render 缓冲的只读视图（`render_buffer.h` 的 `RenderBuffer`）。
pub struct RenderBufferView<'a> {
    blocks: &'a Ring<[f32; BLOCK_SIZE]>,
    spectra: &'a Ring<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    ffts: &'a Ring<FftData>,
}

impl<'a> RenderBufferView<'a> {
    /// 取块（`GetBlock`）。offset=0 为当前对齐块，负值更新。
    pub fn get_block(&self, offset_blocks: isize) -> &'a [f32; BLOCK_SIZE] {
        let position = self
            .blocks
            .offset_index(self.blocks.read(), offset_blocks);
        self.blocks.get(position)
    }

    /// 取功率谱（`Spectrum`）。offset=0 为当前对齐块，正值更旧。
    pub fn spectrum(&self, offset_ffts: isize) -> &'a [f32; FFT_LENGTH_BY_2_PLUS_1] {
        let position = self
            .spectra
            .offset_index(self.spectra.read(), offset_ffts);
        self.spectra.get(position)
    }

    /// FFT 环（`GetFftBuffer`），配合 `position()` 使用。
    pub fn fft_buffer(&self) -> &'a Ring<FftData> {
        self.ffts
    }

    /// 当前对齐位置（`Position()` = fft.read；spectra/ffts 的读写索引必须一致）。
    pub fn position(&self) -> isize {
        debug_assert!(self.spectra.read() == self.ffts.read());
        debug_assert!(self.spectra.write() == self.ffts.write());
        self.ffts.read()
    }

    /// write 与 read 之间的块数（`Headroom`）。
    pub fn headroom(&self) -> isize {
        let size = self.ffts.size() as isize;
        if self.ffts.write() <= self.ffts.read() {
            self.ffts.read() - self.ffts.write()
        } else {
            size - self.ffts.write() + self.ffts.read()
        }
    }

    /// 最近 `num_spectra` 块功率谱之和（`SpectralSum`）。
    pub fn spectral_sum(&self, num_spectra: usize, x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        x2.fill(0.0);
        let mut position = self.spectra.read();
        for _ in 0..num_spectra {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[k] += self.spectra.get(position)[k];
            }
            position = self.spectra.inc_index(position);
        }
    }

    /// 一次遍历同时算两个长度的功率谱和（`SpectralSums`）。
    pub fn spectral_sums(
        &self,
        num_spectra_shorter: usize,
        num_spectra_longer: usize,
        x2_shorter: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        x2_longer: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        debug_assert!(num_spectra_shorter <= num_spectra_longer);
        x2_shorter.fill(0.0);
        let mut position = self.spectra.read();
        for _ in 0..num_spectra_shorter {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2_shorter[k] += self.spectra.get(position)[k];
            }
            position = self.spectra.inc_index(position);
        }
        x2_longer.copy_from_slice(x2_shorter);
        for _ in num_spectra_shorter..num_spectra_longer {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2_longer[k] += self.spectra.get(position)[k];
            }
            position = self.spectra.inc_index(position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T5: 三环读指针方向与延迟语义（total delay = latency + 对齐延迟）。
    #[test]
    fn ring_direction_invariants() {
        let mut buf = RenderDelayBuffer::new();
        let d = 10usize;
        for i in 0..40usize {
            buf.insert(&[i as f32; BLOCK_SIZE]);
            buf.prepare_capture_processing();
        }
        let latency = buf.buffer_latency();
        assert!(buf.align_from_delay(d));
        // ApplyTotalDelay 后的不变量：blocks.read = write − total，spectra.read = write + total
        let total = (latency + d) as isize;
        assert_eq!(
            buf.blocks.read(),
            buf.blocks.offset_index(buf.blocks.write(), -total),
            "blocks 读指针应为 write − (latency+delay)"
        );
        assert_eq!(
            buf.spectra.read(),
            buf.spectra.offset_index(buf.spectra.write(), total),
            "spectra 读指针应为 write + (latency+delay)"
        );
        let view = buf.get_render_buffer();
        assert_eq!(view.position(), buf.ffts.read());
        // GetBlock(0) = total 块前写入的块：insert 先移指针再写，位置 p 存值 p−1，
        // write=40、read=write−total ⟹ 块值应为 read−1。
        let expect_val = (buf.blocks.read() - 1) as f32;
        assert_eq!(view.get_block(0)[0], expect_val);
        assert!(view.headroom() >= 0);
    }

    /// T6: 低速率环逆序写入 + BufferLatency 变化。
    #[test]
    fn low_rate_reversed_insert_and_latency() {
        let mut buf = RenderDelayBuffer::new();
        let block: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32 * 0.05).sin() * 1000.0).collect();
        let mut block_arr = [0.0f32; BLOCK_SIZE];
        block_arr.copy_from_slice(&block);
        // reset() 后 read=write+16；insert 使 write −16 → 未读 32 样本 = 2 块
        buf.insert(&block_arr);
        assert_eq!(buf.buffer_latency(), 2, "仅 insert 后 latency 应为 2");
        buf.prepare_capture_processing();
        assert_eq!(buf.buffer_latency(), 1, "prepare 后 latency 应为 1");

        // 逆序性：write 处 16 个样本 == decimator 直接输出的逆序
        let lr = buf.get_downsampled_render_buffer();
        let w = lr.write() as usize;
        let mut dec = Decimator::new(DELAY_DOWN_SAMPLING_FACTOR);
        let mut ds = [0.0f32; DELAY_SUB_BLOCK_SIZE];
        dec.decimate(&block_arr, &mut ds);
        for k in 0..DELAY_SUB_BLOCK_SIZE {
            assert!(
                (lr.buffer[w + k] - ds[DELAY_SUB_BLOCK_SIZE - 1 - k]).abs() < 1e-4,
                "k={}: ring={} expected rev={}",
                k,
                lr.buffer[w + k],
                ds[DELAY_SUB_BLOCK_SIZE - 1 - k]
            );
        }
    }

    /// 连续插入/消费：underrun 事件与延迟递减。
    #[test]
    fn underrun_behavior() {
        let mut buf = RenderDelayBuffer::new();
        buf.insert(&[100.0; BLOCK_SIZE]);
        assert_eq!(buf.align_from_delay(3), true);
        // 消费多块但不插入 → underrun，delay 递减
        let mut underruns = 0;
        for _ in 0..10 {
            if buf.prepare_capture_processing() == BufferingEvent::RenderUnderrun {
                underruns += 1;
            }
        }
        assert!(underruns > 0);
        assert!(buf.delay().min(3) <= 3);
    }

    /// AlignFromDelay 幂等性：同值不触发变化。
    #[test]
    fn align_idempotent() {
        let mut buf = RenderDelayBuffer::new();
        for i in 0..30 {
            buf.insert(&[(i % 7) as f32 * 500.0; BLOCK_SIZE]);
            buf.prepare_capture_processing();
        }
        assert!(buf.align_from_delay(7));
        assert!(!buf.align_from_delay(7));
        assert!(buf.align_from_delay(9));
    }
}
