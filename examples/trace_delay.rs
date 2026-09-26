use webrtc_linear_aec_rust::block_processor::BlockProcessor;
use webrtc_linear_aec_rust::constants::{BLOCK_SIZE, FRAME_SIZE};

fn main() {
    let mut reader = hound::WavReader::open("tmp/farend_speech.wav").unwrap();
    let far: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32).collect();
    let mut reader = hound::WavReader::open("tmp/nearend_mic.wav").unwrap();
    let near: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32).collect();
    let n = far.len().min(near.len());
    let mut bp = BlockProcessor::new();
    let mut pending_r = Vec::new();
    let mut pending_c = Vec::new();
    let mut blocks = 0usize;
    for f in 0..(n / FRAME_SIZE) {
        pending_r.extend_from_slice(&far[f * FRAME_SIZE..(f + 1) * FRAME_SIZE]);
        pending_c.extend_from_slice(&near[f * FRAME_SIZE..(f + 1) * FRAME_SIZE]);
        while pending_r.len() >= BLOCK_SIZE {
            let rb: [f32; BLOCK_SIZE] = pending_r[..BLOCK_SIZE].try_into().unwrap();
            bp.buffer_render(&rb);
            pending_r.drain(..BLOCK_SIZE);
        }
        while pending_c.len() >= BLOCK_SIZE {
            let mut cb: [f32; BLOCK_SIZE] = pending_c[..BLOCK_SIZE].try_into().unwrap();
            let sat = cb.iter().any(|v| *v >= 32700.0 || *v <= -32700.0);
            bp.process_capture(false, sat, None, &mut cb);
            pending_c.drain(..BLOCK_SIZE);
            blocks += 1;
        }
        if (f + 1) % 100 == 0 {
            let st = bp.echo_remover().aec_state();
            println!("t={:5.2}s 块={} delay={:?} usable={} 初始={} 强块计数={}",
                (f + 1) as f32 * 0.01, blocks, bp.delay_blocks(),
                st.usable_linear_estimate(), st.initial_state_active(),
                st.debug_strong_blocks());
        }
    }
}
