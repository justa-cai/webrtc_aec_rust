//! ERLE 估计器（简化版），对照
//! `modules/audio_processing/aec3/erle_estimator.cc` / `subband_erle_estimator.cc`。
//!
//! 简化策略（见 README 偏差清单）：
//! - `erle[k]` 初值 1；滤波器收敛且非双讲时向 `min(Y2/E2, erle·2)` 以 0.05 平滑爬升；
//! - 钳位 [1, 4.0(低半带)/1.5(高半带)]；`erle_unbounded` 同式，上界 1e5；
//! - 双讲冻结更新；不做上游的 onset 补偿双轨与质量加权。

use crate::constants::{
    ERLE_MAX_H, ERLE_MAX_L, ERLE_MIN, ERLE_UNBOUNDED_MAX, FFT_LENGTH_BY_2,
    FFT_LENGTH_BY_2_PLUS_1,
};

/// ERLE 估计器（简化）。单声道（逐捕获通道一套，本实现仅一套）。
#[derive(Clone, Debug)]
pub struct ErleEstimator {
    erle: [f32; FFT_LENGTH_BY_2_PLUS_1],
    erle_unbounded: [f32; FFT_LENGTH_BY_2_PLUS_1],
    /// 实验开关：只升不降（防双讲边界闪变侵蚀 ERLE → R2 偏大 → 检测失灵）。
    no_downward: bool,
    /// 实验开关：ERLE 上界覆盖（默认 [4.0, 1.5]）。
    max_l_override: f32,
    max_h_override: f32,
}

impl Default for ErleEstimator {
    fn default() -> Self {
        Self::new()
    }
}

fn erle_max(k: usize) -> f32 {
    // 低半带 bins 0..32 → 4.0；高半带 bins 32..65 → 1.5
    if k < FFT_LENGTH_BY_2 / 2 {
        ERLE_MAX_L
    } else {
        ERLE_MAX_H
    }
}

impl ErleEstimator {
    pub fn new() -> Self {
        Self::with_options(false, ERLE_MAX_L, ERLE_MAX_H)
    }

    /// 实验选项构造。
    pub fn with_options(no_downward: bool, max_l: f32, max_h: f32) -> Self {
        Self {
            erle: [ERLE_MIN; FFT_LENGTH_BY_2_PLUS_1],
            erle_unbounded: [ERLE_MIN; FFT_LENGTH_BY_2_PLUS_1],
            no_downward,
            max_l_override: max_l,
            max_h_override: max_h,
        }
    }

    pub fn erle(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.erle
    }

    pub fn erle_unbounded(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.erle_unbounded
    }

    /// 每块更新。`converged` 为滤波器收敛判定；双讲（dominant_nearend）时冻结。
    pub fn update(
        &mut self,
        converged: bool,
        dominant_nearend: bool,
        y2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        e2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        if !converged || dominant_nearend {
            return;
        }
        let max_l = self.max_l_override;
        let max_h = self.max_h_override;
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            // 观测 ERLE；限制每块最多翻倍（防脉冲虚高）
            let obs = y2[k] / (e2[k] + 1.0);
            let cap = if k < FFT_LENGTH_BY_2 / 2 { max_l } else { max_h };
            let mut target = (obs.min(self.erle[k] * 2.0)).clamp(ERLE_MIN, cap);
            if self.no_downward && target < self.erle[k] {
                target = self.erle[k]; // 只升不降
            }
            self.erle[k] += 0.05 * (target - self.erle[k]);
            let target_ub =
                (obs.min(self.erle_unbounded[k] * 2.0)).clamp(ERLE_MIN, ERLE_UNBOUNDED_MAX);
            self.erle_unbounded[k] += 0.05 * (target_ub - self.erle_unbounded[k]);
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 收敛且无双讲时向 Y2/E2 爬升；双讲冻结；钳位生效。
    #[test]
    fn erle_climbs_and_freezes() {
        let mut est = ErleEstimator::new();
        let y2 = [4.0e6f32; FFT_LENGTH_BY_2_PLUS_1];
        let e2 = [1.0e6f32; FFT_LENGTH_BY_2_PLUS_1]; // 真实 ERLE = 4
        // 未收敛：不动
        est.update(false, false, &y2, &e2);
        assert_eq!(est.erle()[16], 1.0);
        // 收敛：爬向 4（低半带），高半带上限 1.5
        for _ in 0..500 {
            est.update(true, false, &y2, &e2);
        }
        assert!((est.erle()[16] - 4.0).abs() < 0.1, "erle={}", est.erle()[16]);
        assert!((est.erle()[48] - 1.5).abs() < 0.05, "erle_h={}", est.erle()[48]);
        assert!(est.erle_unbounded()[16] >= est.erle()[16]);
        // 双讲冻结
        let before = est.erle()[16];
        for _ in 0..100 {
            est.update(true, true, &[1e9; FFT_LENGTH_BY_2_PLUS_1], &[1.0; 65]);
        }
        assert_eq!(est.erle()[16], before);
    }
}
