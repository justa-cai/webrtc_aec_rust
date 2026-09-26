//! 集成测试公共原语：确定性噪声、合成 RIR、卷积。

#![allow(dead_code)]

/// xorshift64* 确定性噪声，输出 [−amp, amp) 无偏白噪声。
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        // [0,2) − 1 → [−1,1)
        ((v >> 33) as f32 / (1u64 << 30) as f32 - 1.0)
    }
    pub fn next_amp(&mut self, amp: f32) -> f32 {
        self.next_f32() * amp
    }
}

/// 合成 RIR：延迟 `delay_samples` + 指数衰减尾（`tail_blocks` 块），
/// 能量归一化到 `gain²`（默认 1，ERL≈0 dB）。
pub fn make_rir(delay_samples: usize, tail_blocks: usize, gain: f64) -> Vec<f64> {
    let len = delay_samples + tail_blocks * 64;
    let mut h = vec![0.0f64; len];
    for k in 0..(tail_blocks * 64) {
        h[delay_samples + k] = (-(k as f64) / (8.0 * 64.0)).exp();
    }
    let energy: f64 = h.iter().map(|v| v * v).sum();
    let scale = gain / energy.sqrt();
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

/// f64 卷积：`x` 为完整历史（含当前块），输出 `out.len()` 个当前样本。
pub fn conv(h: &[f64], x: &[f32], out: &mut [f32]) {
    for n in 0..out.len() {
        let mut acc = 0.0f64;
        for (k, hk) in h.iter().enumerate() {
            let idx = x.len() as isize - out.len() as isize + n as isize - k as isize;
            if idx >= 0 {
                acc += hk * x[idx as usize] as f64;
            }
        }
        out[n] = acc as f32;
    }
}

/// 滑动历史容器：保留 RIR 所需长度（h_len + 一帧余量，保证卷积完整）。
pub struct History {
    buf: Vec<f32>,
    keep: usize,
}

impl History {
    pub fn new(h_len: usize) -> Self {
        Self {
            buf: Vec::new(),
            keep: h_len + 4 * 160,
        }
    }
    pub fn push(&mut self, block: &[f32]) {
        self.buf.extend_from_slice(block);
        if self.buf.len() > self.keep + 4 * 64 {
            let cut = self.buf.len() - self.keep;
            self.buf.drain(..cut);
        }
    }
    pub fn buf(&self) -> &[f32] {
        &self.buf
    }
}
