# ternary-tnn

Ternary neural network layers for Rust. Weights live in {-1, 0, +1}. The forward pass does no floating-point multiplications — only additions, subtractions, and skips.

This is the layer-primitive crate behind the BitNet b1.58 idea: quantize float weights to trits at training time, run inference with integer arithmetic, dequantize once per output neuron with a single scale multiply. Microsoft's 2024 paper showed this matches float16 quality at scale. This crate gives you the building blocks.

---

## The Math

**Weight quantization** (BitNet 1.58-bit scheme):

Given a weight matrix row **w** ∈ ℝⁿ:

```
scale   = mean(|w_i|)                   // L1 mean absolute value
w̃_i    = round(w_i / scale)            // normalize then round
t_i     = clamp(w̃_i, −1, +1)          // project to {−1, 0, +1}
```

The scale is stored as a single f32 per output neuron (not per weight). At inference, the dequantized output is `scale * (t · x)`.

**LUT matmul** — the core trick:

Standard: `y_i = Σ_j w_{ij} · x_j`  (one FMA per element)

Ternary: for each element, `w_{ij}` ∈ {−1, 0, +1}, so:

```
w_{ij} = +1  →  accumulate +x_j
w_{ij} = -1  →  accumulate -x_j
w_{ij} =  0  →  skip entirely
```

Zero multiplications. The lookup table is just a `match` on i8.

**Straight-through estimator (STE)**:

The sign function has derivative 0 almost everywhere, making standard backprop impossible. Bengio et al. (2013) proposed substituting the identity gradient in the region where the latent weight is still "alive":

```
∂L/∂w_latent ≈ ∂L/∂w_ternary · 𝟙[|w_latent| ≤ 1]
```

This crate's `ste_gradient(grad, weight)` returns `grad` if `|weight| ≤ 1`, else `0.0`.

**Convolutions (1D and 2D)** follow the same principle — ternary kernel weights mean each multiply-accumulate becomes an add, subtract, or no-op. Output length formula:

```
out_len = floor((in_len + 2·pad − kernel_size) / stride) + 1
```

---

## Architecture

```
src/lib.rs  (529 LOC, 22 tests)

TernaryActivation           // quantize: f32 → i8 ∈ {-1,0,+1}
  .quantize(x)              // sign function
  .quantize_threshold(x, t) // dead-zone threshold for sparsity

quantize_weights(w)         // flat slice → (Vec<i8>, Vec<f32> scales)
quantize_weight_matrix(w, out, in)  // per-row scale version

TernaryLinear               // fully-connected layer
  .from_float(w, in, out)   // quantize on construction
  .forward(input)           // LUT matmul + scale

TernaryConv1d               // 1-D convolution, ternary kernels
  .output_length(input_len) // dimension arithmetic
  .forward(input, input_len)

TernaryConv2d               // 2-D convolution, ternary kernels
  .output_size(h, w)
  .forward(input, h, w)

lut_matmul(weights, input, rows, cols)  // raw ternary mat-vec
ste_gradient(grad, weight)              // STE scalar
ste_gradient_batch(grads, weights)      // STE over slices
count_ops_ternary(m, n, k)  // additions only: m*n*k
count_ops_float(m, n, k)    // FMAs: 2*m*n*k
```

Memory layout is row-major throughout: `weights[i * in_features + j]` is the weight from input `j` to output neuron `i`.

---

## Quick Start

```toml
[dependencies]
ternary-tnn = "0.1"
```

```rust
use ternary_tnn::{TernaryLinear, lut_matmul, quantize_weights};

// Quantize a float weight matrix you got from training
let float_weights: Vec<f32> = vec![0.8, -1.2, 0.0, 0.4, -0.9, 0.3];
let layer = TernaryLinear::from_float(&float_weights, 3, 2);

// Forward pass: no multiplications
let input = vec![1.0_f32, -0.5, 0.7];
let output = layer.forward(&input);
// output[i] = scales[i] * Σ_j t_{ij} * input[j]

// Or call lut_matmul directly with your own ternary weights
let (trits, _scales) = quantize_weights(&float_weights);
let out = lut_matmul(&trits, &input, 2, 3);
```

---

## Performance

Operation counts for an M×N×K matmul:

| Precision | Operations | Formula |
|-----------|------------|---------|
| float32   | 2·M·N·K    | 1 mul + 1 add per element |
| ternary   | M·N·K      | 1 add-or-sub per element, ~50% skipped |

The `count_ops_*` functions give you exact numbers:

```rust
let (m, n, k) = (768, 768, 512);
println!("float ops : {}", count_ops_float(m, n, k));   // 603,979,776
println!("ternary ops: {}", count_ops_ternary(m, n, k)); // 301,989,888
```

Beyond the op count: ternary weights store at 2 bits/weight vs 32 bits for float32, giving **16× memory reduction** per weight tensor. A 7B parameter model at float32 = 28 GB; at 1.58-bit = ~1.7 GB. The KV-cache savings at runtime follow similarly — see `ternary-llm` for that story.

Energy savings on dedicated hardware (NVIDIA estimates, BitNet b1.58 paper): roughly **10× lower energy per inference** for large batch sizes, because the memory bandwidth bottleneck shrinks proportionally to weight size.

---

## Ecosystem

This crate is the weight-layer primitive. The rest of the stack:

- [`ternary-attention`](https://github.com/OpenClaw/ternary-attention) — scaled dot-product attention with ternary Q/K/V projections and RoPE encoding
- [`ternary-llm`](https://github.com/OpenClaw/ternary-llm) — full transformer block: `TokenEmbedding → TernaryAttentionHead → TernaryFFN → KvCache → TernaryLM`
- [`ternary-grad`](https://github.com/OpenClaw/ternary-grad) — STE-aware optimizers (SGD, Adam), gradient clipping, cosine/step LR schedules for training ternary networks

---

## License

MIT
