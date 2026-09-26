//! 通用环形缓冲，对照 `block_buffer.h` / `fft_buffer.h` / `spectrum_buffer.h` /
//! `downsampled_render_buffer.h`。
//!
//! # 方向语义（照抄源码，不要"统一"）
//!
//! AEC3 中各环的读写指针步进方向不同，**方向翻转即语义**：
//! - blocks 环：`IncWriteIndex`（正向 +1）；
//! - spectra/ffts 环：`DecWriteIndex`（负向 −1）；
//! - 低速率环：write/read 均以 ±`sub_block_size`（16）步进，且入库时数据**逆序**写入。
//!
//! 因此本类型只提供与 C++ 相同的 `inc_index/dec_index/offset_index/inc_write/dec_write/
//! inc_read/dec_read` 原语，由调用方决定步进方向。

/// 环形缓冲 + 读写索引（对应 `BlockBuffer`/`FftBuffer`/`SpectrumBuffer` 的公共骨架）。
#[derive(Clone, Debug)]
pub struct Ring<T> {
    buffer: Vec<T>,
    size: usize,
    write: isize,
    read: isize,
}

impl<T: Clone> Ring<T> {
    pub fn new(size: usize, initial: T) -> Self {
        Self {
            buffer: vec![initial; size],
            size,
            write: 0,
            read: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn write(&self) -> isize {
        self.write
    }
    pub fn read(&self) -> isize {
        self.read
    }
    pub fn set_write(&mut self, i: isize) {
        self.write = i;
    }
    pub fn set_read(&mut self, i: isize) {
        self.read = i;
    }

    pub fn inc_index(&self, index: isize) -> isize {
        if index < self.size as isize - 1 {
            index + 1
        } else {
            0
        }
    }
    pub fn dec_index(&self, index: isize) -> isize {
        if index > 0 {
            index - 1
        } else {
            self.size as isize - 1
        }
    }
    /// `(size + index + offset).rem_euclid(size)`，offset 可为负。
    ///
    /// C++ 原版是 `(size + index + offset) % size` 且由 DCHECK 保证和非负；
    /// 这里用 rem_euclid 对越界 offset 也给出数学上正确的回绕（更安全，
    /// 对所有合法输入与 C++ 语义一致）。
    pub fn offset_index(&self, index: isize, offset: isize) -> isize {
        (self.size as isize + index + offset).rem_euclid(self.size as isize)
    }

    pub fn inc_write(&mut self) {
        self.write = self.inc_index(self.write);
    }
    pub fn dec_write(&mut self) {
        self.write = self.dec_index(self.write);
    }
    pub fn update_write(&mut self, offset: isize) {
        self.write = self.offset_index(self.write, offset);
    }
    pub fn inc_read(&mut self) {
        self.read = self.inc_index(self.read);
    }
    pub fn dec_read(&mut self) {
        self.read = self.dec_index(self.read);
    }
    pub fn update_read(&mut self, offset: isize) {
        self.read = self.offset_index(self.read, offset);
    }

    pub fn get(&self, i: isize) -> &T {
        &self.buffer[i as usize]
    }
    pub fn get_mut(&mut self, i: isize) -> &mut T {
        &mut self.buffer[i as usize]
    }
    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }
}

/// 降采样 render 数据的环形缓冲（`DownsampledRenderBuffer`）。
///
/// 步进粒度为 `sub_block_size`（=16）而非 1；write 在 `InsertBlock` 中
/// `UpdateWriteIndex(-16)`，read 在 `PrepareCaptureProcessing` 中同样 −16。
#[derive(Clone, Debug)]
pub struct DownsampledRenderBuffer {
    pub buffer: Vec<f32>,
    pub size: usize,
    write: isize,
    read: isize,
    sub_block_size: usize,
}

impl DownsampledRenderBuffer {
    pub fn new(size: usize, sub_block_size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            size,
            write: 0,
            read: 0,
            sub_block_size,
        }
    }

    pub fn write(&self) -> isize {
        self.write
    }
    pub fn read(&self) -> isize {
        self.read
    }
    pub fn set_write(&mut self, i: isize) {
        self.write = i;
    }
    pub fn set_read(&mut self, i: isize) {
        self.read = i;
    }
    pub fn sub_block_size(&self) -> usize {
        self.sub_block_size
    }

    pub fn inc_index(&self, index: isize) -> isize {
        if index < self.size as isize - 1 {
            index + 1
        } else {
            0
        }
    }
    pub fn dec_index(&self, index: isize) -> isize {
        if index > 0 {
            index - 1
        } else {
            self.size as isize - 1
        }
    }
    pub fn offset_index(&self, index: isize, offset: isize) -> isize {
        (self.size as isize + index + offset).rem_euclid(self.size as isize)
    }

    pub fn update_write(&mut self, offset: isize) {
        self.write = self.offset_index(self.write, offset);
    }
    pub fn update_read(&mut self, offset: isize) {
        self.read = self.offset_index(self.read, offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_index_semantics() {
        let r: Ring<f32> = Ring::new(10, 0.0);
        assert_eq!(r.inc_index(0), 1);
        assert_eq!(r.inc_index(9), 0);
        assert_eq!(r.dec_index(0), 9);
        assert_eq!(r.dec_index(5), 4);
        assert_eq!(r.offset_index(0, -1), 9);
        assert_eq!(r.offset_index(9, 1), 0);
        assert_eq!(r.offset_index(3, 20), 3); // 整圈
        assert_eq!(r.offset_index(3, -20), 3);
    }

    #[test]
    fn ring_read_write_ops() {
        let mut r: Ring<i32> = Ring::new(4, 0);
        r.set_write(3);
        r.inc_write();
        assert_eq!(r.write(), 0);
        r.dec_write();
        assert_eq!(r.write(), 3);
        *r.get_mut(2) = 42;
        assert_eq!(*r.get(2), 42);
    }

    #[test]
    fn downsampled_buffer_offsets() {
        let mut d = DownsampledRenderBuffer::new(100, 16);
        d.update_write(-16);
        assert_eq!(d.write(), 84);
        d.update_write(-16);
        assert_eq!(d.write(), 68);
        d.set_write(4);
        d.update_read(-16); // read 从 0 出发: (100+0-16) mod 100 = 84
        assert_eq!(d.read(), 84);
    }
}
