//! 精滤波器冲激响应分析，对照
//! `modules/audio_processing/aec3/filter_analyzer.{h,cc}`。
//!
//! - 每 13 块左右完成一次全滤波器扫描（每块推进一个 64-tap 区域）；
//! - 对冲激响应做 ≈600 Hz 最小相位高通预滤波（抑制低频分量干扰峰检测）；
//! - 峰位置（持久候选 + 区域内 argmax）→ 直达路径滤波器延迟（>>6）；
//! - `ConsistentFilterDetector`："滤波器像一个真实冲激响应"判定——排除峰
//!   前后窗后，峰须 >10× 底噪均值且 >2× 次峰；同延迟下活跃 render 累计
//!   >375 块（1.5 s）才判定 consistent。

use crate::constants::{
    ACTIVE_RENDER_BLOCK_ENERGY, BLOCK_SIZE_LOG2, FFT_LENGTH_BY_2, FILTER_ANALYZER_HPF_TAPS,
    NUM_BLOCKS_PER_SECOND,
};
use crate::render_delay_buffer::RenderBufferView;

/// 分析区域（`FilterRegion`）。
#[derive(Clone, Copy, Debug, Default)]
struct FilterRegion {
    start_sample: usize,
    end_sample: usize,
}

/// 持久峰检索：旧峰保持候选，仅在区域内挑战（`FindPeakIndex`）。
fn find_peak_index(filter: &[f32], peak_index_in: usize, start: usize, end: usize) -> usize {
    let mut peak_index_out = peak_index_in;
    let mut max_h2 = filter[peak_index_out] * filter[peak_index_out];
    for k in start..=end {
        let tmp = filter[k] * filter[k];
        if tmp > max_h2 {
            peak_index_out = k;
            max_h2 = tmp;
        }
    }
    peak_index_out
}

/// 一致性检测器（`FilterAnalyzer::ConsistentFilterDetector`）。
#[derive(Clone, Debug)]
struct ConsistentFilterDetector {
    active_render_threshold: f32,
    significant_peak: bool,
    filter_floor_accum: f32,
    filter_secondary_peak: f32,
    filter_floor_low_limit: usize,
    filter_floor_high_limit: usize,
    consistent_estimate_counter: usize,
    consistent_delay_reference: i32,
}

impl ConsistentFilterDetector {
    fn new() -> Self {
        Self {
            active_render_threshold: ACTIVE_RENDER_BLOCK_ENERGY,
            significant_peak: false,
            filter_floor_accum: 0.0,
            filter_secondary_peak: 0.0,
            filter_floor_low_limit: 0,
            filter_floor_high_limit: 0,
            consistent_estimate_counter: 0,
            consistent_delay_reference: -10,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// 一轮扫描到头（start==0）时重置累积器并按峰位设排除窗；
    /// 扫描完成（end==size−1）时做显著峰判定。
    fn detect(
        &mut self,
        filter_to_analyze: &[f32],
        region: &FilterRegion,
        x_block: &[f32],
        peak_index: usize,
        delay_blocks: i32,
    ) -> bool {
        if region.start_sample == 0 {
            self.filter_floor_accum = 0.0;
            self.filter_secondary_peak = 0.0;
            self.filter_floor_low_limit = if peak_index < FFT_LENGTH_BY_2 {
                0
            } else {
                peak_index - FFT_LENGTH_BY_2
            };
            self.filter_floor_high_limit = if peak_index + 128 > filter_to_analyze.len() - 1 {
                0
            } else {
                peak_index + 128
            };
        }

        // 排除窗 [low_limit, high_limit) 之外的样本计入底噪/次峰。
        let mut accum = self.filter_floor_accum;
        let mut secondary = self.filter_secondary_peak;
        let lo_end = (region.end_sample + 1).min(self.filter_floor_low_limit);
        for k in region.start_sample..lo_end {
            let abs_h = filter_to_analyze[k].abs();
            accum += abs_h;
            secondary = secondary.max(abs_h);
        }
        let hi_start = self.filter_floor_high_limit.max(region.start_sample);
        for k in hi_start..=region.end_sample {
            let abs_h = filter_to_analyze[k].abs();
            accum += abs_h;
            secondary = secondary.max(abs_h);
        }
        self.filter_floor_accum = accum;
        self.filter_secondary_peak = secondary;

        if region.end_sample == filter_to_analyze.len() - 1 {
            let denom = (self.filter_floor_low_limit + filter_to_analyze.len()
                - self.filter_floor_high_limit) as f32;
            let filter_floor = self.filter_floor_accum / denom;
            let abs_peak = filter_to_analyze[peak_index].abs();
            self.significant_peak = abs_peak > 10.0 * filter_floor
                && abs_peak > 2.0 * self.filter_secondary_peak;
        }

        if self.significant_peak {
            let x_energy: f32 = x_block.iter().map(|v| v * v).sum();
            let active_render_block = x_energy > self.active_render_threshold;
            if self.consistent_delay_reference == delay_blocks {
                if active_render_block {
                    self.consistent_estimate_counter += 1;
                }
            } else {
                self.consistent_estimate_counter = 0;
                self.consistent_delay_reference = delay_blocks;
            }
        }
        (self.consistent_estimate_counter as f32) > 1.5 * NUM_BLOCKS_PER_SECOND as f32
    }
}

/// 滤波器分析器（`FilterAnalyzer`），单声道。
pub struct FilterAnalyzer {
    default_gain: f32,
    h_highpass: Vec<f32>,
    peak_index: usize,
    gain: f32,
    filter_length_blocks: f32,
    consistent_estimate: bool,
    detector: ConsistentFilterDetector,
    filter_delay_blocks: i32,
    min_filter_delay_blocks: i32,
    region: FilterRegion,
    blocks_since_reset: usize,
}

impl FilterAnalyzer {
    pub fn new() -> Self {
        let mut s = Self {
            default_gain: 1.0, // ep_strength.default_gain
            h_highpass: vec![0.0; crate::constants::time_domain_length(
                crate::constants::FILTER_REFINED_LENGTH_BLOCKS,
            )],
            peak_index: 0,
            gain: 1.0,
            filter_length_blocks: 0.0,
            consistent_estimate: false,
            detector: ConsistentFilterDetector::new(),
            filter_delay_blocks: 0,
            min_filter_delay_blocks: 0,
            region: FilterRegion::default(),
            blocks_since_reset: 0,
        };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.blocks_since_reset = 0;
        self.region = FilterRegion::default();
        self.peak_index = 0;
        self.gain = self.default_gain;
        self.filter_length_blocks = 0.0;
        self.consistent_estimate = false;
        self.detector.reset();
        self.filter_delay_blocks = 0;
        self.min_filter_delay_blocks = 0;
    }

    pub fn min_filter_delay_blocks(&self) -> i32 {
        self.min_filter_delay_blocks
    }

    pub fn consistent_estimate(&self) -> bool {
        self.consistent_estimate
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// 每块更新（`FilterAnalyzer::Update`），返回 (any_filter_consistent, max_gain)。
    pub fn update(&mut self, filter_time_domain: &[f32], render_buffer: &RenderBufferView) {
        self.blocks_since_reset += 1;
        self.set_region_to_analyze(filter_time_domain.len());
        self.analyze_region(filter_time_domain, render_buffer);
        self.min_filter_delay_blocks = self.filter_delay_blocks;
    }

    fn analyze_region(&mut self, filter: &[f32], render_buffer: &RenderBufferView) {
        self.pre_process_filters(filter);

        self.peak_index = self.peak_index.min(self.h_highpass.len() - 1);
        self.peak_index = find_peak_index(
            &self.h_highpass,
            self.peak_index,
            self.region.start_sample,
            self.region.end_sample,
        );
        self.filter_delay_blocks = (self.peak_index >> BLOCK_SIZE_LOG2) as i32;
        self.update_filter_gain();
        self.filter_length_blocks = filter.len() as f32 / FFT_LENGTH_BY_2 as f32;

        let x_block = *render_buffer.get_block(-self.filter_delay_blocks as isize);
        let peak = self.peak_index;
        let delay = self.filter_delay_blocks;
        let region = self.region;
        self.consistent_estimate =
            self.detector
                .detect(&self.h_highpass, &region, &x_block, peak, delay);
    }

    fn update_filter_gain(&mut self) {
        let sufficient_time_to_converge = self.blocks_since_reset > 5 * NUM_BLOCKS_PER_SECOND;
        let abs_peak = self.h_highpass[self.peak_index].abs();
        if sufficient_time_to_converge && self.consistent_estimate {
            self.gain = abs_peak;
        } else if self.gain != 0.0 {
            self.gain = self.gain.max(abs_peak);
        }
        // bounded_erl 默认 false，未移植该分支。
    }

    /// 区域内的高通预滤波（`PreProcessFilters`）。
    fn pre_process_filters(&mut self, filter: &[f32]) {
        self.h_highpass.resize(filter.len(), 0.0);
        for v in self.h_highpass
            .iter_mut()
            .take(self.region.end_sample + 1)
            .skip(self.region.start_sample)
        {
            *v = 0.0;
        }
        let h = FILTER_ANALYZER_HPF_TAPS;
        let start = (h.len() - 1).max(self.region.start_sample);
        for k in start..=self.region.end_sample {
            let mut tmp = self.h_highpass[k];
            for j in 0..h.len() {
                tmp += filter[k - j] * h[j];
            }
            self.h_highpass[k] = tmp;
        }
    }

    /// 区域推进：每块前进 64 tap，越过末端后回到 0（`SetRegionToAnalyze`）。
    fn set_region_to_analyze(&mut self, filter_size: usize) {
        const NUMBER_BLOCKS_TO_UPDATE: usize = 1;
        self.region.start_sample = if self.region.end_sample >= filter_size - 1 {
            0
        } else {
            self.region.end_sample + 1
        };
        self.region.end_sample = (self.region.start_sample
            + NUMBER_BLOCKS_TO_UPDATE * FFT_LENGTH_BY_2
            - 1)
            .min(filter_size - 1);
    }
}

impl Default for FilterAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 峰检索：持久候选 + 区域挑战（旧峰是现任，区域内的峰必须更大才接班）。
    #[test]
    fn peak_finding() {
        let mut h = vec![0.0f32; 832];
        h[100] = 5.0;
        h[600] = 3.0;
        // 区域 [64,127] 含 100 峰
        assert_eq!(find_peak_index(&h, 0, 64, 127), 100);
        // 区域不含更大峰 → 保持旧候选
        assert_eq!(find_peak_index(&h, 100, 128, 191), 100);
        // 区域峰(3.0)小于旧峰(5.0) → 仍是旧峰
        assert_eq!(find_peak_index(&h, 100, 576, 639), 100);
        // 区域峰更大 → 更新
        h[600] = 7.0;
        assert_eq!(find_peak_index(&h, 100, 576, 639), 600);
    }
}
