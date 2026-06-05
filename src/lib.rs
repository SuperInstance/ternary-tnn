//! Ternary Neural Network layers.
//!
//! Weights are quantized to {-1, 0, +1}, enabling multiply-free inference.
//! Includes TernaryLinear, TernaryConv1d/2d, straight-through estimator,
//! LUT-accelerated matmul, and operation-count benchmarks.

// ── Activation / quantization ────────────────────────────────────────────────

/// Sign-function quantizer: maps floats to {-1, 0, +1}.
pub struct TernaryActivation;

impl TernaryActivation {
    #[inline]
    pub fn quantize(x: f32) -> i8 {
        if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
    }

    /// Values whose |x| ≤ threshold map to 0 (sparsity threshold).
    #[inline]
    pub fn quantize_threshold(x: f32, threshold: f32) -> i8 {
        if x.abs() <= threshold { 0 } else { Self::quantize(x) }
    }

    pub fn quantize_batch(xs: &[f32]) -> Vec<i8> {
        xs.iter().map(|&x| Self::quantize(x)).collect()
    }

    pub fn quantize_batch_threshold(xs: &[f32], threshold: f32) -> Vec<i8> {
        xs.iter().map(|&x| Self::quantize_threshold(x, threshold)).collect()
    }
}

// ── Weight quantization ───────────────────────────────────────────────────────

/// Quantize a float weight vector to ternary {-1, 0, +1} plus a single scale.
///
/// Uses the BitNet approach: scale = mean(|w|), then threshold at 0.5.
/// Returns `(ternary_weights, scales)` where each element of `scales` is the
/// per-weight scale — callers typically keep one scale per output neuron row.
pub fn quantize_weights(weights: &[f32]) -> (Vec<i8>, Vec<f32>) {
    if weights.is_empty() {
        return (vec![], vec![]);
    }
    let mean_abs = weights.iter().map(|x| x.abs()).sum::<f32>() / weights.len() as f32;
    let scale = if mean_abs < f32::EPSILON { 1.0 } else { mean_abs };
    let ternary = weights
        .iter()
        .map(|&w| {
            let wn = w / scale;
            if wn > 0.5 { 1i8 } else if wn < -0.5 { -1i8 } else { 0i8 }
        })
        .collect();
    // Return per-element scale (same value) so callers can index easily.
    (ternary, vec![scale; weights.len()])
}

/// Quantize a 2-D weight matrix (out_features × in_features), one scale per row.
pub fn quantize_weight_matrix(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut ternary = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(out_features);
    for row in 0..out_features {
        let row_w = &weights[row * in_features..(row + 1) * in_features];
        let mean_abs = row_w.iter().map(|x| x.abs()).sum::<f32>() / in_features as f32;
        let scale = if mean_abs < f32::EPSILON { 1.0 } else { mean_abs };
        scales.push(scale);
        for &w in row_w {
            let wn = w / scale;
            ternary.push(if wn > 0.5 { 1i8 } else if wn < -0.5 { -1i8 } else { 0i8 });
        }
    }
    (ternary, scales)
}

// ── TernaryLinear ─────────────────────────────────────────────────────────────

/// Fully-connected layer with ternary {-1,0,+1} weights and per-row INT2 scale.
pub struct TernaryLinear {
    pub in_features: usize,
    pub out_features: usize,
    /// Row-major: `weights[i * in_features + j]` is weight from input j to output i.
    pub weights: Vec<i8>,
    /// One scale per output neuron (dequantization multiplier).
    pub scales: Vec<f32>,
}

impl TernaryLinear {
    /// Create a zeroed layer.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            weights: vec![0i8; out_features * in_features],
            scales: vec![1.0f32; out_features],
        }
    }

    /// Quantize a float weight matrix (out × in) into a ternary layer.
    pub fn from_float(weights: &[f32], in_features: usize, out_features: usize) -> Self {
        let (ternary, scales) = quantize_weight_matrix(weights, out_features, in_features);
        Self { in_features, out_features, weights: ternary, scales }
    }

    /// Forward pass: ternary matmul + scale.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features, "input length mismatch");
        (0..self.out_features)
            .map(|i| {
                let row = &self.weights[i * self.in_features..(i + 1) * self.in_features];
                // LUT-style: +1 → add, -1 → subtract, 0 → skip
                let raw: f32 = row.iter().zip(input.iter()).map(|(&w, &x)| match w {
                    1 => x,
                    -1 => -x,
                    _ => 0.0,
                }).sum();
                raw * self.scales[i]
            })
            .collect()
    }
}

// ── TernaryConv1d ─────────────────────────────────────────────────────────────

/// 1-D convolution with ternary kernel weights.
pub struct TernaryConv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    /// `weights[oc * in_channels * kernel_size + ic * kernel_size + k]`
    pub weights: Vec<i8>,
}

impl TernaryConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights: vec![0i8; out_channels * in_channels * kernel_size],
        }
    }

    /// Output length for a given input length (floor division).
    pub fn output_length(&self, input_len: usize) -> usize {
        (input_len + 2 * self.padding).saturating_sub(self.kernel_size) / self.stride + 1
    }

    /// `input` is [in_channels × input_len] row-major.
    /// Returns [out_channels × output_len] row-major.
    pub fn forward(&self, input: &[f32], input_len: usize) -> Vec<f32> {
        let out_len = self.output_length(input_len);
        let mut output = vec![0.0f32; self.out_channels * out_len];
        for oc in 0..self.out_channels {
            for o in 0..out_len {
                let mut sum = 0.0f32;
                for ic in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        let src_pos = o * self.stride + k;
                        // padding: treat out-of-bounds as 0
                        let input_pos = src_pos.checked_sub(self.padding);
                        if let Some(p) = input_pos {
                            if p < input_len {
                                let w = self.weights
                                    [oc * self.in_channels * self.kernel_size
                                        + ic * self.kernel_size
                                        + k];
                                match w {
                                    1 => sum += input[ic * input_len + p],
                                    -1 => sum -= input[ic * input_len + p],
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                output[oc * out_len + o] = sum;
            }
        }
        output
    }
}

// ── TernaryConv2d ─────────────────────────────────────────────────────────────

/// 2-D convolution with ternary kernel weights.
pub struct TernaryConv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
    /// `weights[oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw]`
    pub weights: Vec<i8>,
}

impl TernaryConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            weights: vec![0i8; out_channels * in_channels * kernel_h * kernel_w],
        }
    }

    /// Output spatial dimensions for a given input (h, w).
    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let oh = (h + 2 * self.padding).saturating_sub(self.kernel_h) / self.stride + 1;
        let ow = (w + 2 * self.padding).saturating_sub(self.kernel_w) / self.stride + 1;
        (oh, ow)
    }

    /// `input` is [in_channels × h × w] row-major.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (oh, ow) = self.output_size(h, w);
        let mut output = vec![0.0f32; self.out_channels * oh * ow];
        let ksize = self.kernel_h * self.kernel_w;
        for oc in 0..self.out_channels {
            for r in 0..oh {
                for c in 0..ow {
                    let mut sum = 0.0f32;
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = r * self.stride + kh;
                                let iw = c * self.stride + kw;
                                let ih = ih.checked_sub(self.padding);
                                let iw2 = iw.checked_sub(self.padding);
                                if let (Some(ih), Some(iw)) = (ih, iw2) {
                                    if ih < h && iw < w {
                                        let w_idx = oc * self.in_channels * ksize
                                            + ic * ksize
                                            + kh * self.kernel_w
                                            + kw;
                                        match self.weights[w_idx] {
                                            1 => sum += input[ic * h * w + ih * w + iw],
                                            -1 => sum -= input[ic * h * w + ih * w + iw],
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                    output[oc * oh * ow + r * ow + c] = sum;
                }
            }
        }
        output
    }
}

// ── Straight-Through Estimator ────────────────────────────────────────────────

/// STE gradient: pass gradient through if |w| ≤ 1, zero otherwise.
///
/// During training the sign quantizer has zero derivative everywhere, so we
/// substitute the identity gradient in the bounded region — the "straight
/// through" trick from Bengio et al. (2013).
#[inline]
pub fn ste_gradient(grad_output: f32, weight: f32) -> f32 {
    if weight.abs() <= 1.0 { grad_output } else { 0.0 }
}

pub fn ste_gradient_batch(grad_outputs: &[f32], weights: &[f32]) -> Vec<f32> {
    grad_outputs
        .iter()
        .zip(weights.iter())
        .map(|(&g, &w)| ste_gradient(g, w))
        .collect()
}

// ── LUT-Accelerated MatMul ────────────────────────────────────────────────────

/// Matrix-vector multiply with ternary weights using add/subtract only.
///
/// For each row-vector dot product the weight can be:
///   +1 → accumulate x
///   -1 → accumulate -x
///    0 → skip (contributes nothing)
/// No floating-point multiplications are required.
pub fn lut_matmul(weights: &[i8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(input.len(), cols);
    (0..rows)
        .map(|r| {
            weights[r * cols..(r + 1) * cols]
                .iter()
                .zip(input.iter())
                .map(|(&w, &x)| match w {
                    1 => x,
                    -1 => -x,
                    _ => 0.0,
                })
                .sum()
        })
        .collect()
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

/// Arithmetic operations for an M×N×K ternary matmul (additions only, no muls).
pub fn count_ops_ternary(m: usize, n: usize, k: usize) -> u64 {
    (m * n * k) as u64
}

/// Arithmetic operations for an M×N×K float32 matmul (1 FMA = 1 mul + 1 add).
pub fn count_ops_float(m: usize, n: usize, k: usize) -> u64 {
    2 * (m * n * k) as u64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- TernaryActivation ---

    #[test]
    fn activation_positive_gives_one() {
        assert_eq!(TernaryActivation::quantize(0.5), 1);
        assert_eq!(TernaryActivation::quantize(100.0), 1);
    }

    #[test]
    fn activation_negative_gives_minus_one() {
        assert_eq!(TernaryActivation::quantize(-0.5), -1);
        assert_eq!(TernaryActivation::quantize(-100.0), -1);
    }

    #[test]
    fn activation_zero_gives_zero() {
        assert_eq!(TernaryActivation::quantize(0.0), 0);
    }

    #[test]
    fn activation_threshold_within_band() {
        // |0.1| <= 0.5 → 0
        assert_eq!(TernaryActivation::quantize_threshold(0.1, 0.5), 0);
        assert_eq!(TernaryActivation::quantize_threshold(-0.1, 0.5), 0);
    }

    #[test]
    fn activation_threshold_outside_band() {
        assert_eq!(TernaryActivation::quantize_threshold(1.0, 0.5), 1);
        assert_eq!(TernaryActivation::quantize_threshold(-1.0, 0.5), -1);
    }

    #[test]
    fn activation_batch_length_matches() {
        let xs = vec![1.0, -1.0, 0.0, 2.0, -2.0];
        let out = TernaryActivation::quantize_batch(&xs);
        assert_eq!(out.len(), xs.len());
    }

    // --- quantize_weights ---

    #[test]
    fn quantize_weights_only_ternary_values() {
        let w = vec![0.1, -0.5, 1.2, -1.5, 0.0, 0.8];
        let (t, _) = quantize_weights(&w);
        for &v in &t {
            assert!(v == -1 || v == 0 || v == 1, "unexpected value {v}");
        }
    }

    #[test]
    fn quantize_weights_scale_positive() {
        let w = vec![1.0, 2.0, -1.0, -2.0];
        let (_, scales) = quantize_weights(&w);
        for &s in &scales {
            assert!(s > 0.0);
        }
    }

    // --- TernaryLinear ---

    #[test]
    fn linear_forward_output_length() {
        let layer = TernaryLinear::new(8, 4);
        let input = vec![1.0f32; 8];
        assert_eq!(layer.forward(&input).len(), 4);
    }

    #[test]
    fn linear_known_weights_known_output() {
        // One output neuron, weights = [1, 0, -1], scale = 1.0
        let mut layer = TernaryLinear::new(3, 1);
        layer.weights = vec![1i8, 0, -1];
        layer.scales = vec![1.0];
        let out = layer.forward(&[2.0, 5.0, 3.0]);
        // 1*2 + 0*5 + (-1)*3 = 2 - 3 = -1
        assert!((out[0] - (-1.0)).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn linear_from_float_roundtrip() {
        let float_w: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.5).collect();
        let layer = TernaryLinear::from_float(&float_w, 4, 3);
        assert_eq!(layer.weights.len(), 12);
        assert_eq!(layer.scales.len(), 3);
    }

    // --- LUT matmul ---

    #[test]
    fn lut_matmul_matches_naive() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, 0, 1];
        let input: Vec<f32> = vec![1.0, 2.0, 3.0];
        let out = lut_matmul(&weights, &input, 2, 3);
        // row0: 1*1 + (-1)*2 + 0*3 = -1
        // row1: 1*1 + 0*2 + 1*3 = 4
        assert!((out[0] - (-1.0)).abs() < 1e-6, "row0 = {}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-6, "row1 = {}", out[1]);
    }

    #[test]
    fn lut_matmul_all_zero_weights() {
        let weights = vec![0i8; 6];
        let input = vec![1.0f32, 2.0, 3.0];
        let out = lut_matmul(&weights, &input, 2, 3);
        for &v in &out {
            assert!(v.abs() < 1e-9);
        }
    }

    // --- STE ---

    #[test]
    fn ste_passes_gradient_inside() {
        let g = ste_gradient(0.5, 0.8);
        assert!((g - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ste_blocks_gradient_outside() {
        let g = ste_gradient(0.5, 1.5);
        assert!(g.abs() < 1e-9);
    }

    #[test]
    fn ste_batch_length_matches() {
        let grads = vec![1.0, 2.0, 3.0];
        let weights = vec![0.5, 1.5, -0.5];
        let out = ste_gradient_batch(&grads, &weights);
        assert_eq!(out.len(), 3);
        // only index 0 and 2 should pass
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!(out[1].abs() < 1e-9);
        assert!((out[2] - 3.0).abs() < 1e-9);
    }

    // --- Benchmark ---

    #[test]
    fn ternary_ops_fewer_than_float() {
        let (m, n, k) = (64, 64, 64);
        assert!(count_ops_ternary(m, n, k) < count_ops_float(m, n, k));
        // Specifically: ternary uses exactly half the ops (only additions, no muls)
        assert_eq!(count_ops_float(m, n, k), 2 * count_ops_ternary(m, n, k));
    }

    // --- TernaryConv1d ---

    #[test]
    fn conv1d_output_length_no_padding() {
        let c = TernaryConv1d::new(1, 1, 3, 1, 0);
        assert_eq!(c.output_length(10), 8); // (10 - 3) / 1 + 1 = 8
    }

    #[test]
    fn conv1d_output_length_with_padding() {
        let c = TernaryConv1d::new(1, 1, 3, 1, 1);
        assert_eq!(c.output_length(10), 10); // (10 + 2 - 3) / 1 + 1 = 10
    }

    #[test]
    fn conv1d_forward_shape() {
        let mut c = TernaryConv1d::new(2, 3, 3, 1, 0);
        // Fill with +1 weights for a known result
        c.weights = vec![1i8; 3 * 2 * 3];
        let input = vec![1.0f32; 2 * 10]; // 2 channels × 10 time steps
        let out = c.forward(&input, 10);
        // out_len = 8, out_channels = 3
        assert_eq!(out.len(), 3 * 8);
    }

    // --- TernaryConv2d ---

    #[test]
    fn conv2d_output_size_no_padding() {
        let c = TernaryConv2d::new(1, 1, 3, 3, 1, 0);
        assert_eq!(c.output_size(8, 8), (6, 6)); // (8-3)/1+1 = 6
    }

    #[test]
    fn conv2d_forward_shape() {
        let c = TernaryConv2d::new(1, 2, 3, 3, 1, 0);
        let input = vec![1.0f32; 1 * 8 * 8];
        let out = c.forward(&input, 8, 8);
        let (oh, ow) = c.output_size(8, 8);
        assert_eq!(out.len(), 2 * oh * ow);
    }
}
