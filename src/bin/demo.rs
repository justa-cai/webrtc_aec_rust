//! 演示 CLI。
//!
//! 用法：
//! - `demo --selftest`：内部合成信号（8 s WGN render + 100 ms 延迟指数衰减 RIR），
//!   打印时延检出、锁定时间、分段 ERLE、RTF
//! - `demo far.wav near.wav out.wav [--linear-out path.wav]`：
//!   WAV 模式（16 kHz 单声道 s16），far=远端参考，near=麦克风，out=线性 AEC 输出

use std::time::Instant;
use webrtc_linear_aec_rust::constants::FRAME_SIZE;
use webrtc_linear_aec_rust::EchoCanceller;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--selftest" {
        selftest();
        return;
    }
    if args.len() >= 4 {
        let far = &args[1];
        let near = &args[2];
        let out = &args[3];
        let linear_out = args
            .iter()
            .position(|a| a == "--linear-out")
            .and_then(|i| args.get(i + 1))
            .cloned();
        run_wav(far, near, out, linear_out.as_deref());
        return;
    }
    eprintln!("用法:");
    eprintln!("  demo --selftest                       合成信号自测");
    eprintln!("  demo far.wav near.wav out.wav [--linear-out path.wav]");
    std::process::exit(1);
}

/// 确定性白噪声（与测试一致的 xorshift64*）。
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as f32 / (1u64 << 30) as f32 - 1.0
    }
}

/// 合成 RIR：延迟 + 指数衰减尾，单位能量。
fn make_rir(delay_samples: usize, tail_blocks: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; delay_samples + tail_blocks * 64];
    for k in 0..tail_blocks * 64 {
        h[delay_samples + k] = (-(k as f64) / 512.0).exp();
    }
    let energy: f64 = h.iter().map(|v| v * v).sum();
    let scale = 1.0 / energy.sqrt();
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

fn selftest() {
    println!("=== webrtc-linear-aec-rust 自测 ===");
    println!("场景: 8 s WGN render (RMS≈1732) + 100 ms 延迟/单位能量指数衰减 RIR\n");

    let delay_samples = 1600usize;
    let h = make_rir(delay_samples, 4);
    let mut rng = Rng(0xB00B5);
    let mut aec = EchoCanceller::new();
    let mut hist: Vec<f32> = Vec::new();
    let keep = h.len() + 640;
    let n_frames = 800usize;

    let start = Instant::now();
    let (mut sy, mut se) = (0.0f64, 0.0f64);
    let mut lock_frame: Option<usize> = None;
    let mut usable_frame: Option<usize> = None;
    let mut final_delay: Option<usize> = None;

    for f in 0..n_frames {
        let mut rf = [0.0f32; FRAME_SIZE];
        for v in rf.iter_mut() {
            *v = rng.next() * 3000.0;
        }
        hist.extend_from_slice(&rf);
        if hist.len() > keep {
            let cut = hist.len() - keep;
            hist.drain(..cut);
        }
        let mut cf = [0.0f32; FRAME_SIZE];
        for n in 0..FRAME_SIZE {
            let mut acc = 0.0f64;
            for (k, hk) in h.iter().enumerate() {
                let i = hist.len() as isize - FRAME_SIZE as isize + n as isize - k as isize;
                if i >= 0 {
                    acc += hk * hist[i as usize] as f64;
                }
            }
            cf[n] = acc as f32;
        }
        let y2: f64 = cf.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        aec.push_render_frame(&rf);
        let m = aec.process_capture_frame(&mut cf);
        let e2: f64 = cf.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        sy += y2;
        se += e2;

        if lock_frame.is_none() && m.delay_blocks.is_some() {
            lock_frame = Some(f);
        }
        if usable_frame.is_none() && m.usable_linear_estimate {
            usable_frame = Some(f);
        }
        if m.delay_blocks.is_some() {
            final_delay = m.delay_blocks;
        }
        if (f + 1) % 100 == 0 {
            println!(
                "  t={:4.1}s  分段累计 ERLE = {:6.1} dB",
                (f + 1) as f64 / 100.0,
                10.0 * (sy / se).log10()
            );
            sy = 0.0;
            se = 0.0;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n时延检出: 首个估计 @ {:.2}s, 最终 {} 块 (真值 {} 块)",
        lock_frame.map(|f| f as f64 * 0.01).unwrap_or(-1.0),
        final_delay.unwrap_or(0),
        delay_samples / 64);
    println!("线性估计可用 (UsableLinearEstimate): {:.2}s",
        usable_frame.map(|f| f as f64 * 0.01).unwrap_or(-1.0));
    println!("总 ERLE: {:.1} dB", aec.erle_db().unwrap_or(0.0));
    println!("RTF: {:.4}（{} s 音频 / {:.3} s 计算）",
        elapsed / (n_frames as f64 * 0.01),
        n_frames as f64 * 0.01,
        elapsed);
}

fn run_wav(far: &str, near: &str, out: &str, linear_out: Option<&str>) {
    let far_spec = hound::WavReader::open(far).unwrap_or_else(|e| panic!("打开 {} 失败: {}", far, e));
    let near_spec =
        hound::WavReader::open(near).unwrap_or_else(|e| panic!("打开 {} 失败: {}", near, e));
    assert_eq!(
        far_spec.spec().sample_rate,
        16000,
        "仅支持 16 kHz（far 为 {} Hz）",
        far_spec.spec().sample_rate
    );
    assert_eq!(near_spec.spec().sample_rate, 16000, "仅支持 16 kHz");
    assert_eq!(far_spec.spec().channels, 1, "仅支持单声道 far");
    assert_eq!(near_spec.spec().channels, 1, "仅支持单声道 near");

    let far: Vec<f32> = far_spec
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32)
        .collect();
    let near: Vec<f32> = near_spec
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32)
        .collect();
    let n = far.len().min(near.len());
    println!("输入: {} 样本（{:.2} s），far/near 取较短者", n, n as f64 / 16000.0);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out, spec).expect("创建输出 WAV 失败");
    let mut linear_writer = linear_out.map(|p| {
        hound::WavWriter::create(p, spec)
            .unwrap_or_else(|e| panic!("创建 {} 失败: {}", p, e))
    });

    let mut aec = EchoCanceller::new();
    let start = Instant::now();
    for chunk in far[..n].chunks(FRAME_SIZE) {
        let mut rf = [0.0f32; FRAME_SIZE];
        rf[..chunk.len()].copy_from_slice(chunk);
        // 保持驱动节奏：capture 与 far 同步推进（按 far 的帧位置取 near）
        let pos = (chunk.as_ptr() as usize - far.as_ptr() as usize) / 4;
        let mut cf = [0.0f32; FRAME_SIZE];
        let end = (pos + FRAME_SIZE).min(n);
        if pos < n {
            let m = (end - pos).min(FRAME_SIZE);
            cf[..m].copy_from_slice(&near[pos..end]);
        }
        aec.push_render_frame(&rf);
        if let Some(lw) = linear_writer.as_mut() {
            let mut lo = [0.0f32; FRAME_SIZE];
            let metrics = aec.process_capture_frame_with_linear_output(&mut cf, &mut lo);
            let _ = metrics;
            for v in lo.iter() {
                lw.write_sample((v.clamp(-32768.0, 32767.0)) as i16).unwrap();
            }
        } else {
            aec.process_capture_frame(&mut cf);
        }
        for v in cf.iter() {
            writer
                .write_sample((v.clamp(-32768.0, 32767.0)) as i16)
                .unwrap();
        }
    }
    writer.finalize().unwrap();
    if let Some(lw) = linear_writer {
        lw.finalize().unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("完成: {} ({}{}), 总 ERLE {:.1} dB, RTF {:.4}",
        out,
        linear_out.unwrap_or(""),
        if linear_out.is_some() { "（含 linear-out）" } else { "" },
        aec.erle_db().unwrap_or(0.0),
        elapsed / (n as f64 / 16000.0));
}
