//! WebRTC AEC3 线性部分（线性 AEC + 时延估计）的 Rust 参考实现。
//!
//! 对照源码仓库 `../webrtc`（Chromium tip, commit a3553f1）逐文件移植：
//! - 线性 AEC：`modules/audio_processing/aec3/{adaptive_fir_filter,subtractor,echo_remover,
//!   refined_filter_update_gain,coarse_filter_update_gain}.cc`
//! - 时延估计：`modules/audio_processing/aec3/{matched_filter,matched_filter_lag_aggregator,
//!   echo_path_delay_estimator,render_delay_controller,decimator}.cc`
//!
//! 原理文档见 `../webrtc/tmp/linear-aec-principles.md`（README 中有章节映射）。
//!
//! 信号域：16 kHz、单声道；块 64 样本（4 ms）、FFT 128 点、65 个 bin、250 块/秒。

pub mod aec_state;
pub mod adaptive_fir_filter;
pub mod block_processor;
pub mod clockdrift_detector;
pub mod coarse_filter_update_gain;
pub mod constants;
pub mod decimator;
pub mod delay_estimate;
pub mod echo_canceller;
pub mod echo_path_delay_estimator;
pub mod echo_path_variability;
pub mod echo_remover;
pub mod fft;
pub mod fft_data;
pub mod filter_analyzer;
pub mod matched_filter;
pub mod matched_filter_lag_aggregator;
pub mod refined_filter_update_gain;
pub mod render_delay_buffer;
pub mod render_delay_controller;
pub mod render_signal_analyzer;
pub mod ring;
pub mod subtractor;
pub mod subtractor_output;

pub use block_processor::RenderEvent;
pub use constants::FRAME_SIZE;
pub use echo_canceller::{EchoCanceller, FrameMetrics};
