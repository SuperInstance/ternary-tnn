# PLUG_AND_PLAY — Tnn

> Ternary neural network layers: {-1, 0, +1} weights

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-tnn = { git = "https://github.com/SuperInstance/ternary-tnn" }
```

Use in your code:

```rust
use ternary_tnn::{TernaryLinear, TernaryActivation};

let mut layer = TernaryLinear::new(128, 64);
let out = layer.forward(&input);
```

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
