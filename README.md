# webrtc-linear-aec-rust

WebRTC AEC3 **线性部分**（线性 AEC + 时延估计）的 Rust 参考实现。

对照源码：`../webrtc`（Chromium tip, commit `a3553f1`）的 `modules/audio_processing/aec3/`，
按文件逐个移植；原理讲解见 `docs/linear-aec-principles.md`。

## 范围

- ✅ 线性 AEC：分区频域自适应滤波（PBFDAF）、粗/精双滤波器、循环轮转时域约束、
  失配整体缩放、coarse←refined 系数复制、变步长（H_error 递推 + 频域泄漏）
- ✅ 时延估计：4× 降采样（椭圆低通+Butterworth 高通）、5×512-tap 匹配滤波器组（NLMS）、
  滑动直方图双门限投票、前回声（pre-echo）对齐、时钟漂移检测、缓冲迟滞
- ✅ 组装：render 多级缓冲与延迟对齐、AecState 收敛门控、FilterAnalyzer、
  10 ms 帧顶层 API
- ❌ 不含：非线性残余抑制（SuppressionGain/CNG/ResidualEchoEstimator）、多声道、
  >16 kHz 高带、SIMD、透明模式、ML 残余估计、外部时延估计路径、ERLE/ERL 完整估计器

信号域：**16 kHz、单声道**；块 64 样本（4 ms）、FFT 128 点、65 bin、250 块/s——与 AEC3 相同。

## 构建 / 测试 / 运行

```bash
cargo test                              # 43 单元 + 6 集成测试
cargo test --release                    # 集成测试含 RTF<0.1 断言（实测 ≈0.025）
cargo run --release --bin demo -- --selftest
cargo run --release --bin demo -- far.wav near.wav out.wav [--linear-out lin.wav]
```

WAV 模式要求 16 kHz 单声道 s16。

## 快速理解：一条数据流

```
render 帧(160) ─► RenderDelayBuffer ─► 三环形缓冲(块/FFT/谱) + 低速率降采样环
                        │                        (blocks 正向步进, 谱/FFT 反向)
capture 帧(160) ─► BlockProcessor
                        ├─ RenderDelayController::get_delay
                        │    └─ EchoPathDelayEstimator: capture 降采样 →
                        │       MatchedFilter 组(5×512 NLMS) → 直方图投票 →
                        │       pre-echo 候选 → ×4 → 块换算(>>6) + 迟滞
                        ├─ AlignFromDelay → ApplyTotalDelay(移动读指针)
                        │   （延迟变化 ⟹ EchoPathVariability ⟹ 滤波器全量复位）
                        └─ EchoRemover(线性部分)
                             ├─ RenderSignalAnalyzer(窄带检测/掩蔽)
                             ├─ Subtractor: 精滤波(H_error 变步长+泄漏+约束轮转)
                             │            + 粗滤波(定步长, 连差 5 块复制精系数)
                             │            + 失配缩放 + PredictionError
                             ├─ UseRefinedOutput 判据 + 30 样本交叉淡化
                             ├─ Y2/E2/S2_linear 谱
                             └─ AecState: 收敛/发散判定 + InitialState(2.5s) +
                                FilterAnalyzer(峰位/一致性) + UsableLinearEstimate 门控
```

## 模块 ↔ WebRTC 源文件 ↔ 原理文档映射

| Rust 模块 | WebRTC 源文件（modules/audio_processing/ 下） | 原理文档章节 |
|---|---|---|
| `constants.rs` | `aec3/aec3_common.h`, `api/audio/echo_canceller3_config.h` | — |
| `fft.rs`, `fft_data.rs` | `aec3/aec3_fft.{h,cc}`, `aec3/fft_data.h`（窗表照抄；FFT 后端换 rustfft） | §4.2 |
| `ring.rs` | `aec3/{block,fft,spectrum,downsampled_render}_buffer.h` | §4.3 |
| `decimator.rs` | `aec3/decimator.cc`, `utility/cascaded_biquad_filter.cc` | §5.7 |
| `render_delay_buffer.rs` | `aec3/render_delay_buffer.cc`, `aec3/render_buffer.{h,cc}` | §5.13 |
| `matched_filter.rs` | `aec3/matched_filter.cc`（标量核） | §5.5/5.6/5.10 |
| `matched_filter_lag_aggregator.rs` | `aec3/matched_filter_lag_aggregator.cc` | §5.8/5.10 |
| `clockdrift_detector.rs` | `aec3/clockdrift_detector.cc` | §5.12 |
| `echo_path_delay_estimator.rs` | `aec3/echo_path_delay_estimator.cc` | §5.14 |
| `render_delay_controller.rs` | `aec3/render_delay_controller.cc` | §5.11 |
| `adaptive_fir_filter.rs` | `aec3/adaptive_fir_filter.cc`, `aec3/adaptive_fir_filter_erl.cc` | §4.3–4.7 |
| `refined_filter_update_gain.rs` | `aec3/refined_filter_update_gain.cc` | §6.3/6.4 |
| `coarse_filter_update_gain.rs` | `aec3/coarse_filter_update_gain.cc` | §3.3 |
| `subtractor.rs`, `subtractor_output.rs` | `aec3/subtractor.cc`, `aec3/subtractor_output.cc` | §7 |
| `render_signal_analyzer.rs` | `aec3/render_signal_analyzer.cc`（窄带部分） | §6.6 |
| `filter_analyzer.rs` | `aec3/filter_analyzer.cc` | — |
| `aec_state.rs` | `aec3/aec_state.cc`, `aec3/subtractor_output_analyzer.cc` | §8 |
| `echo_remover.rs` | `aec3/echo_remover.cc`（线性部分） | §7.3/7.4 |
| `block_processor.rs` | `aec3/block_processor.cc` | §2 |
| `echo_canceller.rs` | `aec3/echo_canceller3.cc`（帧调度；FIFO 组块等价于 FrameBlocker） | §1 |

原理文档：[docs/linear-aec-principles.md](docs/linear-aec-principles.md)（章节号见上表右列）。

## FFT 缩放约定（与 WebRTC 数值等价）

WebRTC 的 Ooura：前向不缩放，`InverseFft(Fft(x)) = 64·x`，各调用点乘 1/64。
本实现：rustfft 前向不缩放、逆向含 1/128，**往返严格相等**——因此 X/H/G/S/E/Y
与 WebRTC 逐数组一致，且全程无需 1/64 因子：

- 预测误差：`e = y − ifft(S)[64..128]`
- 时域约束：`ifft → 清零 [64..128) → fft`

H 的绝对尺度被自适应吸收；`H2`/`erl` 只用于相对比较（泄漏项、峰/底噪比值），尺度无关。

## 端到端行为基线（`--selftest`，合成场景）

| 指标 | 数值 |
|---|---|
| 时延检出（100 ms 真值） | 0.22 s 首个估计；报告 24 块（真值 25 块，headroom 32 样本 + 结构性 −1 块，回声落在滤波器内部 ✓） |
| UsableLinearEstimate | 0.42 s 起为 true |
| ERLE（分段） | 1s:5.7 → 2s:30 → 4s:116 → 稳态 ≈133 dB（合成无噪线性回声） |
| RTF（release） | ≈0.021（单线程标量） |

## 已知简化 / 偏差（与上游逐项对照）

| 项 | 说明 |
|---|---|
| 非线性抑制 | 未移植（上游 `SuppressionGain`/`SuppressionFilter`/CNG/`ResidualEchoEstimator`）。双讲残留与非线性失真残余在真实产品中由该层处理 |
| 多声道 | 结构按单声道写死（上游多声道 = 每捕获通道一组滤波器 + 通道混音选择） |
| `render_activity_`、API 抖动跟踪 | 上游仅供残余回声估计/日志，省略并注释 |
| `IdentifyStrongNarrowBandComponent` | 上游仅供抑制器，省略（窄带**计数器**与掩蔽保留） |
| ERLE/ERL 估计器、饱和回声检测、混响模型 | 上游仅指标/抑制器输入，桩化或省略 |
| 透明模式 | 恒不激活（`transparent=false`），不影响 `UsableLinearEstimate` 语义 |
| 外部时延路径 | `SetAudioBufferDelay`/`AlignFromExternalDelay` 未移植（对应 `use_external_delay_estimator`） |
| FFT 后端 | rustfft 复数 FFT（实输入零虚部）替代 Ooura 实数 FFT；窗表从 `aec3_fft.cc` 照抄 |
| 帧调度 | 160 样本样本 FIFO 组块，等价于上游 subframe/FrameBlocker 机制（每 2 帧 5 块） |
| API 级增益变化标志 | 恒 false（上游由应用层传入） |

## 测试覆盖（49 项）

单元（43）：FFT 往返/DFT 对照/PaddedFft 等价/窗表；环形方向与逆序写入/latency；
降采样通带增益/阻带衰减/稳定性；匹配滤波收敛（lag∈{10,400,1000}）/前回声扫描；
直方图双门限/pre-echo 候选/并列取小；迟滞；时钟漂移模式；约束尾部清零/长度渐变/
Scale/SetFilter；失配估计器；增益保护与 H_error 钳位；ERLE 固定对齐（0/100 ms）；
窄带检测/掩蔽；峰检索；门控时序。

集成（6，公共 API）：端到端 ERLE≥30 dB + 时延锁定 ≤1.5 s + 门控时序；
多时延检测精度（50/100/250/400 ms）；延迟跳变（100→250 ms）≤1.5 s 检出 +
4–7 s 窗 ERLE≥20 dB；饱和门控；近端透传（残余抑制 ≥20 dB + 保真比 >0.6）；
性能 RTF<0.1（release）。

## License

BSD-3-Clause（与 WebRTC 上游一致）。
