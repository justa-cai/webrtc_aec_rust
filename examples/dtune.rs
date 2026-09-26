//! 双讲调优实验：参数扫描（E1）+ 近端掩蔽（E3），输出指标表与对比 WAV。
//!
//! 指标（真实样本无干净近端参考，用代理）：
//! - level_db：输出/麦克风电平比（越高=保近端越好，但可能含回声残留）
//! - echo_leak：输出与远端的归一化互相关峰值（越低=回声残留越少）

use webrtc_linear_aec_rust::constants::{SuppressorConfig, FRAME_SIZE};
use webrtc_linear_aec_rust::EchoCanceller;
use webrtc_linear_aec_rust::simple_ns::{NsLevel, SimpleNs};

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

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    reader.samples::<i16>().map(|s| s.unwrap() as f32).collect()
}

/// 跑一组配置，返回 (输出, 逐秒电平比, 全程回声残留互相关)
fn run(far: &[f32], near: &[f32], cfg: SuppressorConfig, ns: Option<NsLevel>) -> (Vec<f32>, Vec<f64>, f64) {
    let mut aec = EchoCanceller::with_suppressor_config(cfg);
    let mut ns_proc = SimpleNs::new(ns.unwrap_or(NsLevel::Moderate));
    let n = far.len().min(near.len());
    let mut out = Vec::with_capacity(n);
    // NS 的 WOLA 滤波器组固有 96 样本延迟：流式去延迟
    let mut delay_buf: Vec<f32> = if ns.is_some() { vec![0.0; 96] } else { Vec::new() };
    let frames = n / FRAME_SIZE;
    for f in 0..frames {
        let mut rf = [0f32; FRAME_SIZE];
        let mut cf = [0f32; FRAME_SIZE];
        rf.copy_from_slice(&far[f * FRAME_SIZE..(f + 1) * FRAME_SIZE]);
        cf.copy_from_slice(&near[f * FRAME_SIZE..(f + 1) * FRAME_SIZE]);
        aec.push_render_frame(&rf);
        aec.process_capture_frame(&mut cf);
        if let Some(level) = ns {
            ns_proc.process(&mut cf);
            delay_buf.extend_from_slice(&cf);
            let drained: Vec<f32> = delay_buf.drain(..FRAME_SIZE).collect();
            out.extend_from_slice(&drained);
        } else {
            out.extend_from_slice(&cf);
        }
    }
    // 逐秒电平比
    let mut levels = Vec::new();
    for s in 0..10 {
        let (a, b) = (s * 16000, (s + 1) * 16000);
        let y2: f64 = near[a..b].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let o2: f64 = out[a..b].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        levels.push(10.0 * (o2 / (y2 + 1e-9)).log10());
    }
    // 回声残留：输出 vs 远端的互相关峰（扫 0..1000ms 延迟）
    let mut best = 0f64;
    for lag in (0..16000).step_by(160) {
        let mut num = 0f64;
        let mut de = 0f64;
        let mut df = 0f64;
        for i in 0..(16000 * 8) {
            let o = out[i + lag] as f64;
            let fv = far[i] as f64;
            num += o * fv;
            de += o * o;
            df += fv * fv;
        }
        let corr = num / ((de * df).sqrt() + 1e-9);
        best = best.max(corr);
    }
    (out, levels, best)
}

fn main() {
    let far = read_wav("tmp/farend_speech.wav");
    let near = read_wav("tmp/nearend_mic.wav");

    // 实验组
    let mut groups: Vec<(&str, SuppressorConfig)> = Vec::new();
    groups.push(("baseline(默认)", SuppressorConfig::default()));

    let mut agg = SuppressorConfig::default();
    // 激进模式：低频保护全开 + 掩蔽 α 降 + 过减加深（牺牲保真换压制）
    agg.nearend_masker = true;
    agg.nearend_masker_alpha = 0.3;
    agg.nearend_tuning.max_dec_factor_lf = 1.0;      // 去掉低频下降限速
    agg.normal_tuning.max_dec_factor_lf = 1.0;
    agg.last_lf_smoothing_band = 0;                   // 低频平滑保护只留 bin0
    agg.nearend_tuning.lf.emr_transparent = 0.1;      // LF emr 底线加深
    agg.normal_tuning.lf.emr_transparent = 0.1;
    agg.erle_max_l = 10.0;
    agg.erle_max_h = 4.0;
    groups.push(("激进低频", agg));

    let mut agg2 = agg;
    agg2.nearend_masker = false;                      // 连掩蔽也关：最大压制
    agg2.dominant_nearend_detection.use_unbounded_echo_spectrum = false;
    agg2.dominant_nearend_detection.enr_threshold = 0.5;
    groups.push(("激进无掩蔽", agg2));

    let mut ga = SuppressorConfig::default();
    ga.dominant_nearend_detection.use_unbounded_echo_spectrum = false;
    groups.push(("A 有界R2检测", ga));

    let mut gb = SuppressorConfig::default();
    gb.erle_no_downward = true;
    groups.push(("B ERLE只升", gb));

    let mut gc = SuppressorConfig::default();
    gc.erle_max_l = 10.0;
    gc.erle_max_h = 4.0;
    groups.push(("C ERLE界10/4", gc));

    let mut gab = SuppressorConfig::default();
    gab.dominant_nearend_detection.use_unbounded_echo_spectrum = false;
    gab.erle_no_downward = true;
    groups.push(("AB", gab));

    let mut gab2 = SuppressorConfig::default();
    gab2.dominant_nearend_detection.use_unbounded_echo_spectrum = false;
    gab2.erle_no_downward = true;
    gab2.dominant_nearend_detection.enr_threshold = 0.5;
    groups.push(("AB+enr0.5", gab2));

    let mut full = SuppressorConfig::default();
    full.dominant_nearend_detection.use_unbounded_echo_spectrum = false;
    full.erle_no_downward = true;
    full.erle_max_l = 10.0;
    full.erle_max_h = 4.0;
    full.dominant_nearend_detection.enr_threshold = 0.5;
    full.nearend_tuning.lf.enr_transparent = 1.29;
    full.nearend_tuning.lf.enr_suppress = 1.3;
    full.nearend_masker = true;
    groups.push(("ABC+L1L2+掩蔽", full));

    let mut g1 = SuppressorConfig::default();
    g1.dominant_nearend_detection.enr_threshold = 0.5;
    groups.push(("L1a enr=0.5", g1));

    let mut g2 = SuppressorConfig::default();
    g2.dominant_nearend_detection.enr_threshold = 0.75;
    groups.push(("L1b enr=0.75", g2));

    let mut g3 = SuppressorConfig::default();
    g3.nearend_tuning.lf.enr_transparent = 1.29;
    g3.nearend_tuning.lf.enr_suppress = 1.3;
    groups.push(("L2 lf(1.29,1.3)", g3));

    let mut g4 = SuppressorConfig::default();
    g4.dominant_nearend_detection.enr_threshold = 0.5;
    g4.nearend_tuning.lf.enr_transparent = 1.29;
    g4.nearend_tuning.lf.enr_suppress = 1.3;
    groups.push(("L1+L2 组合", g4));

    let mut g5 = SuppressorConfig::default();
    g5.nearend_masker = true;
    groups.push(("L3 近端掩蔽", g5));

    let mut gns = SuppressorConfig::default();
    gns.nearend_masker = true;
    groups.push(("L3+NS12dB", gns));

    let mut gns2 = SuppressorConfig::default();
    gns2.nearend_masker = true;
    groups.push(("L3+NS18dB High", gns2));

    let mut gdeep = SuppressorConfig::default();
    gdeep.nearend_masker = true;
    // E2: 非双讲段加深（render_limit 64→16）
    // 注：echo_audibility 未进 SuppressorConfig，此处经由保守 tuning 近似
    gdeep.normal_tuning.hf.enr_transparent = 0.05;
    gdeep.normal_tuning.hf.enr_suppress = 0.07;
    groups.push(("L3+加深", gdeep));

    let mut gall = SuppressorConfig::default();
    gall.nearend_masker = true;
    gall.normal_tuning.hf.enr_transparent = 0.05;
    gall.normal_tuning.hf.enr_suppress = 0.07;
    groups.push(("L3+NS+加深", gall));

    let mut ga = SuppressorConfig::default();
    ga.nearend_masker = true;
    ga.nearend_masker_alpha = 0.6;
    groups.push(("L3a0.6", ga));

    let mut gb = SuppressorConfig::default();
    gb.nearend_masker = true;
    gb.nearend_masker_alpha = 0.35;
    groups.push(("L3a0.35", gb));

    let mut g5b = SuppressorConfig::default();
    g5b.nearend_masker = true;
    g5b.nearend_tuning.hf.emr_transparent = 0.5;
    g5b.normal_tuning.hf.emr_transparent = 0.5;
    groups.push(("L3+emr0.5", g5b));

    let mut g6 = SuppressorConfig::default();
    g6.dominant_nearend_detection.enr_threshold = 0.5;
    g6.nearend_tuning.lf.enr_transparent = 1.29;
    g6.nearend_tuning.lf.enr_suppress = 1.3;
    g6.nearend_masker = true;
    g6.dominant_nearend_detection.trigger_threshold = 6;
    g6.dominant_nearend_detection.hold_duration = 100;
    groups.push(("全家桶", g6));

    println!("{:<16} {:>7} {:>7} {:>7} {:>7} {:>8}", "配置", "秒2", "秒3", "秒6", "秒9", "回声残留");
    println!("{}", "-".repeat(60));
    let mut outputs = Vec::new();
    for (name, cfg) in groups {
        let use_ns = name.contains("NS");
        let lvl = if name.contains("High") { NsLevel::High } else { NsLevel::Moderate };
        let (out, levels, leak) = run(&far, &near, cfg, if use_ns { Some(lvl) } else { None });
        println!(
            "{:<16} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>8.3}",
            name,
            levels[2],
            levels[3],
            levels[6],
            levels[9],
            leak
        );
        outputs.push((name, out));
    }

    // 代表配置合成多通道试听：mic / baseline / L1+L2 / 全家桶
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    for (name, out) in &outputs {
        let fname = format!("tmp/dtune_{}.wav", name.split('(').next().unwrap().split_whitespace().next().unwrap());
        let mut w = hound::WavWriter::create(&fname, spec).unwrap();
        for v in out.iter() {
            w.write_sample((v.clamp(-32768.0, 32767.0)) as i16).unwrap();
        }
        w.finalize().unwrap();
    }
    println!("\n各组 WAV 已输出到 tmp/dtune_*.wav");

    // 多通道对比
    use std::io::Write;
    let picks = ["baseline(默认)", "L3 近端掩蔽", "AB", "ABC+L1L2+掩蔽"];
    let n = near.len();
    let mut writer = hound::WavWriter::create("tmp/dtune_compare.wav", hound::WavSpec {
        channels: (1 + picks.len()) as u16,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }).unwrap();
    for i in 0..n {
        writer.write_sample((near[i].clamp(-32768.0, 32767.0)) as i16).unwrap();
        for p in &picks {
            let o = outputs.iter().find(|(n, _)| n == p).unwrap().1[i];
            writer.write_sample((o.clamp(-32768.0, 32767.0)) as i16).unwrap();
        }
    }
    writer.finalize().unwrap();
    println!("多通道对比: tmp/dtune_compare.wav [1]=mic [2]=baseline [3]=L3掩蔽 [4]=AB [5]=全家桶");
    
}
