//! lag 聚合（多帧直方图投票），对照
//! `modules/audio_processing/aec3/matched_filter_lag_aggregator.{h,cc}`。
//!
//! - `HighestPeakAggregator`：250 项滑动直方图（O(1) 众数），双门限输出
//!   {coarse, refined}，`significant_candidate_found` 为单向锁存。
//! - `PreEchoLagAggregator`：把前回声 lag 量化到块再做直方图投票，
//!   前 500 次更新对越晚的块施加 0.7 递减惩罚（优先相信更早的回声起点）。

use crate::constants::{
    DELAY_HEADROOM_SAMPLES, DELAY_DOWN_SAMPLING_FACTOR, DELAY_SELECTION_THRESHOLD_CONVERGED,
    DELAY_SELECTION_THRESHOLD_INITIAL, LAG_HISTOGRAM_WINDOW, PRE_ECHO_PENALIZATION,
    PRE_ECHO_PENALIZATION_UPDATES,
};
use crate::delay_estimate::{DelayEstimate, DelayQuality};
use crate::matched_filter::LagEstimate;

/// `GetDownSamplingBlockSizeLog2`（matched_filter_lag_aggregator.cc）：
/// kBlockSizeLog2 − log2(down_sampling_factor) = 6 − 2 = 4。
fn down_sampling_block_size_log2() -> u32 {
    let mut f = DELAY_DOWN_SAMPLING_FACTOR >> 1;
    let mut log2 = 0u32;
    while f > 0 {
        log2 += 1;
        f >>= 1;
    }
    if 6 > log2 {
        6 - log2
    } else {
        0
    }
}

/// 滑动直方图投票（`MatchedFilterLagAggregator::HighestPeakAggregator`）。
pub struct HighestPeakAggregator {
    histogram: Vec<i32>,
    histogram_data: [usize; LAG_HISTOGRAM_WINDOW],
    histogram_data_index: usize,
    candidate: usize,
}

impl HighestPeakAggregator {
    pub fn new(max_filter_lag: usize) -> Self {
        Self {
            histogram: vec![0; max_filter_lag + 1],
            histogram_data: [0; LAG_HISTOGRAM_WINDOW],
            histogram_data_index: 0,
            candidate: 0,
        }
    }

    pub fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data = [0; LAG_HISTOGRAM_WINDOW];
        self.histogram_data_index = 0;
    }

    pub fn histogram(&self) -> &[i32] {
        &self.histogram
    }

    pub fn candidate(&self) -> usize {
        self.candidate
    }

    /// O(1) 滑动窗口众数更新：移除最老、加入最新、重取众数。
    pub fn aggregate(&mut self, lag: usize) {
        debug_assert!(self.histogram.len() > self.histogram_data[self.histogram_data_index]);
        let old = self.histogram_data[self.histogram_data_index];
        self.histogram[old] -= 1;
        self.histogram_data[self.histogram_data_index] = lag;
        self.histogram[lag] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % LAG_HISTOGRAM_WINDOW;
        // 并列时取最小 bin（std::max_element 语义）。
        self.candidate = self
            .histogram
            .iter()
            .enumerate()
            .max_by_key(|(i, v)| (*v, i32::MAX - *i as i32))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
}

/// 前回声聚合器（`MatchedFilterLagAggregator::PreEchoLagAggregator`）。
pub struct PreEchoLagAggregator {
    block_size_log2: u32,
    histogram: Vec<i32>,
    histogram_data: [isize; LAG_HISTOGRAM_WINDOW],
    histogram_data_index: usize,
    number_updates: i32,
    pre_echo_candidate: usize,
}

/// 未更新的直方图槽位标记（`kPreEchoHistogramDataNotUpdated`）。
const NOT_UPDATED: isize = -1;

impl PreEchoLagAggregator {
    pub fn new(max_filter_lag: usize) -> Self {
        Self {
            block_size_log2: down_sampling_block_size_log2(),
            histogram: vec![0; ((max_filter_lag + 1) * DELAY_DOWN_SAMPLING_FACTOR) >> 6],
            histogram_data: [NOT_UPDATED; LAG_HISTOGRAM_WINDOW],
            histogram_data_index: 0,
            number_updates: 0,
            pre_echo_candidate: 0,
        }
    }

    pub fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data = [NOT_UPDATED; LAG_HISTOGRAM_WINDOW];
        self.histogram_data_index = 0;
        self.pre_echo_candidate = 0;
        self.number_updates = 0;
    }

    pub fn pre_echo_candidate(&self) -> usize {
        self.pre_echo_candidate
    }

    pub fn aggregate(&mut self, pre_echo_lag: usize) {
        let mut pre_echo_block_size = (pre_echo_lag >> self.block_size_log2) as isize;
        debug_assert!(pre_echo_block_size >= 0 && pre_echo_block_size < self.histogram.len() as isize);
        pre_echo_block_size = pre_echo_block_size.clamp(0, self.histogram.len() as isize - 1);

        // 移除最老的一项（未更新过的槽位跳过）。
        if self.histogram_data[self.histogram_data_index] != NOT_UPDATED {
            self.histogram[self.histogram_data[self.histogram_data_index] as usize] -= 1;
        }
        self.histogram_data[self.histogram_data_index] = pre_echo_block_size;
        self.histogram[pre_echo_block_size as usize] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % LAG_HISTOGRAM_WINDOW;

        let mut candidate: usize = 0;
        if self.number_updates < PRE_ECHO_PENALIZATION_UPDATES {
            self.number_updates += 1;
            // 前 2 秒：按 32-bin 组做 0.7 递减惩罚，优先更早的回声起点。
            // 注意：末尾不足 32 的组被静默忽略（照抄源码循环条件）。
            let mut penalization: f32 = 1.0;
            let mut max_histogram_value = -1.0f32;
            let mut it = 0usize;
            while self.histogram.len() - it >= 32 {
                let group = &self.histogram[it..it + 32];
                let (group_max_idx, group_max) = group
                    .iter()
                    .enumerate()
                    .max_by_key(|(i, v)| (**v, i32::MAX - *i as i32))
                    .unwrap();
                let weighted = *group_max as f32 * penalization;
                if weighted > max_histogram_value {
                    max_histogram_value = weighted;
                    candidate = it + group_max_idx;
                }
                penalization *= PRE_ECHO_PENALIZATION;
                it += 32;
            }
        } else {
            candidate = self
                .histogram
                .iter()
                .enumerate()
                .max_by_key(|(i, v)| (**v, i32::MAX - *i as i32))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        self.pre_echo_candidate = candidate << self.block_size_log2;
    }
}

/// lag 聚合器（`MatchedFilterLagAggregator`）。
pub struct MatchedFilterLagAggregator {
    thresholds: (i32, i32), // (initial, converged)
    headroom: usize,        // 32/4 = 8（降采样样本）
    highest_peak_aggregator: HighestPeakAggregator,
    pre_echo_lag_aggregator: Option<PreEchoLagAggregator>,
    significant_candidate_found: bool,
}

impl MatchedFilterLagAggregator {
    pub fn new(max_filter_lag: usize) -> Self {
        Self {
            thresholds: (
                DELAY_SELECTION_THRESHOLD_INITIAL,
                DELAY_SELECTION_THRESHOLD_CONVERGED,
            ),
            headroom: DELAY_HEADROOM_SAMPLES / DELAY_DOWN_SAMPLING_FACTOR,
            highest_peak_aggregator: HighestPeakAggregator::new(max_filter_lag),
            pre_echo_lag_aggregator: Some(PreEchoLagAggregator::new(max_filter_lag)),
            significant_candidate_found: false,
        }
    }

    pub fn reset(&mut self, hard_reset: bool) {
        self.highest_peak_aggregator.reset();
        if let Some(pre) = self.pre_echo_lag_aggregator.as_mut() {
            pre.reset();
        }
        if hard_reset {
            self.significant_candidate_found = false;
        }
    }

    pub fn reliable_delay_found(&self) -> bool {
        self.significant_candidate_found
    }

    pub fn get_delay_at_highest_peak(&self) -> usize {
        self.highest_peak_aggregator.candidate()
    }

    /// 聚合一次 lag 估计；通过门限时输出 `DelayEstimate`（降采样样本单位）。
    pub fn aggregate(&mut self, lag_estimate: Option<LagEstimate>) -> Option<DelayEstimate> {
        if let Some(est) = lag_estimate {
            if let Some(pre) = self.pre_echo_lag_aggregator.as_mut() {
                pre.aggregate((est.pre_echo_lag as isize - self.headroom as isize).max(0) as usize);
            }

            self.highest_peak_aggregator
                .aggregate((est.lag as isize - self.headroom as isize).max(0) as usize);
            let candidate = self.highest_peak_aggregator.candidate();
            let count = self.highest_peak_aggregator.histogram()[candidate];
            self.significant_candidate_found = self.significant_candidate_found
                || count > self.thresholds.1;
            if count > self.thresholds.1
                || (count > self.thresholds.0 && !self.significant_candidate_found)
            {
                let quality = if self.significant_candidate_found {
                    DelayQuality::Refined
                } else {
                    DelayQuality::Coarse
                };
                let reported_delay = match self.pre_echo_lag_aggregator.as_ref() {
                    Some(pre) => pre.pre_echo_candidate(),
                    None => candidate,
                };
                return Some(DelayEstimate::new(quality, reported_delay));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_lag() -> usize {
        crate::constants::DELAY_NUM_FILTERS * crate::constants::MATCHED_FILTER_INTRA_LAG_SHIFT
            + crate::constants::MATCHED_FILTER_SIZE
    }

    /// T8: 直方图投票双门限与滑动窗口消退。
    #[test]
    fn histogram_voting_thresholds() {
        let mut agg = MatchedFilterLagAggregator::new(max_lag());
        // 前 5 次命中 lag=40：计数 5 未过 initial(>5) → 无输出
        for _ in 0..5 {
            assert!(agg.aggregate(Some(LagEstimate::new(40 + 8, 40 + 8))).is_none());
        }
        // 第 6 次：计数 6 > 5 → coarse
        let out = agg.aggregate(Some(LagEstimate::new(40 + 8, 40 + 8)));
        assert_eq!(
            out.map(|d| d.quality),
            Some(DelayQuality::Coarse),
            "第 6 次应输出 Coarse"
        );
        // 继续命中到计数 21 > 20 → refined
        let mut got_refined = false;
        for i in 0..20 {
            let out = agg.aggregate(Some(LagEstimate::new(40 + 8, 40 + 8)));
            if let Some(d) = out {
                if d.quality == DelayQuality::Refined {
                    got_refined = true;
                    // 计数从 6 涨到 21 的那次（i = 15）翻 Refined
                    assert!(i >= 14, "refined 过早出现于 i={}", i);
                    break;
                }
            }
        }
        assert!(got_refined, "应出现 Refined");
        assert!(agg.reliable_delay_found());

        // 滑动窗口：灌入 250 个不同 lag 后，lag=40 的计数被挤出 → 停止输出该候选
        let mut none_count = 0;
        for i in 0..LAG_HISTOGRAM_WINDOW {
            let out = agg.aggregate(Some(LagEstimate::new(
                100 + 8 + (i % 50),
                100 + 8 + (i % 50),
            )));
            if out.is_none() {
                none_count += 1;
            }
        }
        assert!(none_count > 0, "旧候选被滑窗挤出后应停止输出");
    }

    /// 前回声候选为报告值（detect_pre_echo 默认开）。
    #[test]
    fn pre_echo_candidate_reported() {
        let mut agg = MatchedFilterLagAggregator::new(max_lag());
        // pre_echo_lag = 64+8（headroom 扣除后 64 → 块 4）
        for _ in 0..8 {
            let out = agg.aggregate(Some(LagEstimate::new(100 + 8, 64 + 8)));
            if let Some(d) = out {
                // 报告的是 pre-echo 候选（块量化：4 << 4 = 64），非峰值 100
                assert_eq!(d.delay, 64);
                return;
            }
        }
        panic!("应输出 pre-echo 候选");
    }

    /// 并列取最小 bin。
    #[test]
    fn tie_takes_lowest_bin() {
        let mut agg = HighestPeakAggregator::new(100);
        agg.aggregate(5);
        agg.aggregate(9);
        agg.aggregate(5);
        agg.aggregate(9);
        assert_eq!(agg.candidate(), 5);
    }

    /// 块尺寸对数换算：ds4 → 4。
    #[test]
    fn block_size_log2() {
        assert_eq!(down_sampling_block_size_log2(), 4);
    }
}
