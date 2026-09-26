// AEC 多轨同步播放器 —— Web Audio API，零依赖
"use strict";

const TRACKS = [
  { id: "far",      name: "远端参考",   color: "#ffb454", file: "audio/far.ogg" },
  { id: "mic",      name: "近端麦克风", color: "#ff6b81", file: "audio/mic.ogg" },
  { id: "linear",   name: "线性输出",   color: "#4f8cff", file: "audio/linear.ogg" },
  { id: "nonlinear",name: "非线性输出", color: "#37d4a0", file: "audio/nonlinear.ogg" },
];

const ctx = new (window.AudioContext || window.webkitAudioContext)();
const state = {
  buffers: new Map(),   // id -> AudioBuffer
  nodes: new Map(),     // id -> {src, gain}
  playing: false,
  offset: 0,            // 当前播放位置（秒）
  startedAt: 0,         // ctx.currentTime 起点
  duration: 0,
  muted: new Set(),
  solo: null,
};

// ---------- 加载 ----------
async function load() {
  await Promise.all(TRACKS.map(async (t) => {
    const res = await fetch(t.file);
    const buf = await res.arrayBuffer();
    state.buffers.set(t.id, await ctx.decodeAudioData(buf));
  }));
  state.duration = Math.min(...TRACKS.map((t) => state.buffers.get(t.id).duration));
  buildUI();
  document.getElementById("play-btn").disabled = false;
  document.getElementById("play-btn").textContent = "播放";
  updateTime(0);
}

// ---------- UI 构建 ----------
function buildUI() {
  const wrap = document.getElementById("tracks");
  TRACKS.forEach((t) => {
    const el = document.createElement("div");
    el.className = "track";
    el.innerHTML = `
      <div class="track-controls">
        <div class="track-name" style="color:${t.color}">${t.name}</div>
        <div class="track-tags">
          <button class="btn" data-act="mute">静音</button>
          <button class="btn" data-act="solo">独奏</button>
        </div>
      </div>
      <div class="track-wave"><canvas></canvas></div>`;
    wrap.appendChild(el);

    const muteBtn = el.querySelector('[data-act="mute"]');
    const soloBtn = el.querySelector('[data-act="solo"]');
    muteBtn.addEventListener("click", () => {
      state.muted.has(t.id) ? state.muted.delete(t.id) : state.muted.add(t.id);
      muteBtn.classList.toggle("active-mute", state.muted.has(t.id));
      applyGains();
    });
    soloBtn.addEventListener("click", () => {
      state.solo = state.solo === t.id ? null : t.id;
      TRACKS.forEach((o) => {
        const b = wrap.querySelector(`#${o.id}-row .btn[data-act="solo"]`);
      });
      el.querySelectorAll(".btn")[1].classList.toggle("active-solo", state.solo === t.id);
      applyGains();
    });
    el.id = `${t.id}-row`;

    const cv = el.querySelector("canvas");
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect();
      seek(((e.clientX - r.left) / r.width) * state.duration);
    });
    drawWave(cv, state.buffers.get(t.id), t.color, 0);
    t.canvas = cv;
  });
  window.addEventListener("resize", () => {
    TRACKS.forEach((t) => drawWave(t.canvas, state.buffers.get(t.id), t.color, position()));
  });
}

// ---------- 增益（mute/solo）----------
function applyGains() {
  TRACKS.forEach((t) => {
    const n = state.nodes.get(t.id);
    if (!n) return;
    const audible = state.solo ? state.solo === t.id : !state.muted.has(t.id);
    n.gain.gain.setTargetAtTime(audible ? 1 : 0, ctx.currentTime, 0.01);
  });
}

// ---------- 播放控制 ----------
function startAll(offset) {
  const when = ctx.currentTime + 0.03; // 统一起始时刻，保证采样级同步
  TRACKS.forEach((t) => {
    const src = ctx.createBufferSource();
    src.buffer = state.buffers.get(t.id);
    const gain = ctx.createGain();
    src.connect(gain).connect(ctx.destination);
    src.start(when, offset);
    state.nodes.set(t.id, { src, gain });
  });
  applyGains();
  state.startedAt = when;
  state.offset = offset;
}
function stopAll() {
  state.nodes.forEach(({ src }) => { try { src.stop(); } catch (e) {} });
  state.nodes.clear();
}
function position() {
  if (!state.playing) return state.offset;
  return Math.min(state.duration, state.offset + (ctx.currentTime - state.startedAt));
}
function play() {
  if (state.playing || position() >= state.duration - 0.01) { if (position() >= state.duration - 0.01) { state.offset = 0; } else { return; } }
  ctx.resume();
  startAll(state.offset);
  state.playing = true;
  setBtn();
}
function pause() {
  if (!state.playing) return;
  state.offset = position();
  stopAll();
  state.playing = false;
  setBtn();
}
function seek(t) {
  t = Math.max(0, Math.min(state.duration - 0.01, t));
  if (state.playing) { stopAll(); startAll(t); }
  else { state.offset = t; }
  updateTime(t);
}
function setBtn() {
  const b = document.getElementById("play-btn");
  b.textContent = state.playing ? "暂停" : "播放";
  b.classList.toggle("playing", state.playing);
}

// ---------- 时间显示与渲染循环 ----------
const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function updateTime(t) {
  document.getElementById("time-label").textContent = `${fmt(t)} / ${fmt(state.duration)}`;
}
function frame() {
  if (state.playing) {
    const p = position();
    updateTime(p);
    TRACKS.forEach((t) => drawWave(t.canvas, state.buffers.get(t.id), t.color, p));
    if (p >= state.duration - 0.01) { pause(); }
  }
  requestAnimationFrame(frame);
}

// ---------- 波形绘制（min/max 峰值 + 进度着色）----------
function drawWave(canvas, buffer, color, progress) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const data = buffer.getChannelData(0);
  const spp = data.length / w; // 每像素样本数
  const mid = h / 2;
  const progX = (progress / buffer.duration) * w;

  for (let x = 0; x < w; x++) {
    let mn = 1, mx = -1;
    const i0 = Math.floor(x * spp), i1 = Math.min(data.length, Math.ceil((x + 1) * spp));
    for (let i = i0; i < i1; i++) { if (data[i] < mn) mn = data[i]; if (data[i] > mx) mx = data[i]; }
    const y0 = mid - mx * mid * 0.92, y1 = mid - mn * mid * 0.92;
    g.fillStyle = x <= progX ? color : "#3a4059";
    g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  // 播放头
  g.fillStyle = "rgba(255,255,255,0.85)";
  g.fillRect(progX, 0, 1.5, h);
}

// ---------- 绑定 ----------
document.getElementById("play-btn").addEventListener("click", () => state.playing ? pause() : play());
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); state.playing ? pause() : play(); }
});
window.addEventListener("resize", () => TRACKS.forEach((t) => t.canvas && drawWave(t.canvas, state.buffers.get(t.id), t.color, position())));

load().then(() => requestAnimationFrame(frame)).catch((e) => {
  document.getElementById("play-btn").textContent = "加载失败";
  console.error(e);
});
