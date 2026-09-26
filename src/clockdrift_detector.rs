//! 时钟漂移检测，对照 `modules/audio_processing/aec3/clockdrift_detector.{h,cc}`。
//!
//! 识别延迟估计的规律性单调漂移（±1/±2/±3 步长模式），两级判定：
//! probable（初步怀疑）→ verified（确认）。连续 30 s 稳定则复位。
//!
//! 注意：本移植中该结果为信息位，不影响线性路径决策（与上游一致——
//! 漂移只用于抑制器策略切换）。

use crate::constants::CLOCKDRIFT_STABLE_BLOCKS;

/// 漂移等级（`ClockdriftDetector::Level`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    None,
    Probable,
    Verified,
}

/// 时钟漂移检测器（`ClockdriftDetector`）。
#[derive(Clone, Debug)]
pub struct ClockdriftDetector {
    delay_history: [i32; 3],
    stability_counter: i32,
    level: Level,
}

impl Default for ClockdriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockdriftDetector {
    pub fn new() -> Self {
        Self {
            delay_history: [0; 3],
            stability_counter: 0,
            level: Level::None,
        }
    }

    pub fn level(&self) -> Level {
        self.level
    }

    /// 用新的延迟估计更新（`ClockdriftDetector::Update`）。
    pub fn update(&mut self, delay_estimate: i32) {
        if delay_estimate == self.delay_history[0] {
            // 稳定 7500 块（30 s）后复位漂移等级。
            self.stability_counter += 1;
            if self.stability_counter > CLOCKDRIFT_STABLE_BLOCKS {
                self.level = Level::None;
            }
            return;
        }
        self.stability_counter = 0;

        let d1 = self.delay_history[0] - delay_estimate;
        let d2 = self.delay_history[1] - delay_estimate;
        let d3 = self.delay_history[2] - delay_estimate;

        // 正向漂移模式（远端时钟偏慢）：[x−3], x−2, x−1, x 或次序互换。
        let probable_drift_up = (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
        let drift_up = probable_drift_up && d3 == -3;

        // 负向漂移模式：[x+3], x+2, x+1, x 或次序互换。
        let probable_drift_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
        let drift_down = probable_drift_down && d3 == 3;

        if drift_up || drift_down {
            self.level = Level::Verified;
        } else if (probable_drift_up || probable_drift_down) && self.level == Level::None {
            self.level = Level::Probable;
        }

        self.delay_history[2] = self.delay_history[1];
        self.delay_history[1] = self.delay_history[0];
        self.delay_history[0] = delay_estimate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T24a: 单调递减模式 → Verified。
    #[test]
    fn drift_up_verified() {
        let mut d = ClockdriftDetector::new();
        // 序列 x, x−1, x−2, x−3
        d.update(100);
        d.update(100);
        d.update(99);
        d.update(98);
        d.update(97);
        assert_eq!(d.level(), Level::Verified);
    }

    /// T24b: 稳定 7501 次后回到 None。
    #[test]
    fn stable_resets() {
        let mut d = ClockdriftDetector::new();
        d.update(100);
        d.update(99);
        d.update(98);
        d.update(97);
        assert_eq!(d.level(), Level::Verified);
        for _ in 0..7501 {
            d.update(97);
        }
        assert_eq!(d.level(), Level::None);
    }

    /// 无规律变化不触发。
    #[test]
    fn random_walk_no_drift() {
        let mut d = ClockdriftDetector::new();
        for v in [100, 130, 95, 140, 90, 128] {
            d.update(v);
        }
        assert_eq!(d.level(), Level::None);
    }
}
