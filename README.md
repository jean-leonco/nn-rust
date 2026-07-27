# nn-rust

`nn-rust` is a small neural network library written in Rust. It implements forward propagation, backpropagation, and matrix operations without a high-level machine learning framework.

## Features

- Sequential models with a typed builder
- Dense layers
- ReLU and sigmoid activations
- Dropout
- Softmax with cross-entropy
- SGD
- Model serialization
- MNIST loading and example executables

## Implementation notes

Models compile into a linear operation plan. Parameters and temporary values use contiguous arenas. A training session allocates its working memory once and reuses it across batches.

Dense layers delegate matrix multiplication to CBLAS. They account for most training time as networks become wider. The smaller element-wise operations work on contiguous slices so LLVM can auto-vectorize them. The exponential approximation and dropout generator use portable SIMD directly.

Sigmoid and softmax use the Schraudolph exponential approximation. This trades some numerical accuracy for lower computation cost. Dropout uses a counter-based Philox generator. It derives masks from the seed, operation range, and training step without storing mutable state for every value.

Portable SIMD currently requires the nightly Rust toolchain.

## Build

Install a C compiler, Fortran compiler, and `make`. The Linux build compiles a static OpenBLAS dependency. macOS uses Accelerate.

```sh
cargo build --release
```

## Train and predict

Run the training with:

```sh
cargo run --release --bin train
```

The executable writes `relu_model` and `sigmoid_model` in the current directory.

Run inference on `test_image.png`:

```sh
cargo run --release --bin predict
```

The main model construction API is:

```rust
use nn_rust::{model::SequentialModel, ops::Initialization};

let model = SequentialModel::builder()
    .input(784)
    .dense(128, Initialization::He)
    .relu()
    .dropout(0.2)?
    .dense(10, Initialization::He)
    .softmax()
    .build();

# Ok::<(), Box<dyn std::error::Error>>(())
```

## Performance

Build with `--release`. The Linux runs set `OPENBLAS_NUM_THREADS=8` to match the eight physical cores. The remaining operations run on the calling thread. Accelerate manages its own execution on macOS.

### Workloads

| Executable    | Network                            | Dense layers | Parameters | Epochs |
| ------------- | ---------------------------------- | -----------: | ---------: | -----: |
| `train`       | ReLU: 784 → 128 → 64 → 10          |            3 |    109,386 |     50 |
| `train`       | Sigmoid: 784 → 256 → 64 → 10       |            3 |    218,058 |     50 |
| `train_large` | ReLU: 784 → 2048 → 1024 → 512 → 10 |            4 |  4,235,786 |     20 |

The `train` time covers both listed networks.

### Training results

| System                  | Executable    |    Time | Peak memory |
| ----------------------- | ------------- | ------: | ----------: |
| Ryzen 7 9700X, OpenBLAS | `train`       |  6.53 s |    76.3 MiB |
| Apple M3, Accelerate    | `train`       |  8.82 s |    65.9 MiB |
| Ryzen 7 9700X, OpenBLAS | `train_large` | 20.90 s |   115.0 MiB |
| Apple M3, Accelerate    | `train_large` | 29.90 s |   104.1 MiB |

The Ryzen completed `train` 1.35 times faster than the M3. It completed `train_large` 1.43 times faster. The Linux process used about eight cores in both runs. The larger GEMMs keep those OpenBLAS workers busy for more of the run.

The large network has about 13 times as many parameters as both smaller networks combined. It took 3.2 times longer on Linux and 3.4 times longer on macOS, but it ran for 20 epochs instead of 50. The times therefore do not scale directly with parameter count.

Peak memory rose by about 39 MiB on each system for `train_large`. This similar increase reflects the larger parameter and session arenas. macOS used 10 to 11 MiB less peak memory, but took longer for both workloads.

On Linux, `train_large` reached 2.3 instructions per cycle, compared with 1.7 for `train`. Its L1 data-cache miss rate rose from 15.0% to 16.7%. The wider GEMMs achieved higher instruction throughput despite the added cache pressure.

### Learning results

| Network    | Final train loss | Final train accuracy | Validation loss | Validation accuracy |
| ---------- | ---------------: | -------------------: | --------------: | ------------------: |
| ReLU       |           0.0522 |               98.31% |          0.0697 |              98.09% |
| Sigmoid    |           0.1192 |               96.45% |          0.0969 |              97.03% |
| Large ReLU |           0.2237 |               93.57% |          0.1867 |              94.54% |

The ReLU network converged faster and reached higher accuracy with half as many parameters as the sigmoid network. ReLU is also cheaper to evaluate. It reduces to comparisons and stores, while sigmoid still evaluates an approximate exponential despite its SIMD implementation. The sigmoid network needed more width and a learning rate of 0.2, compared with 0.05 for ReLU.

The ReLU training loss continued to fall late in the run, while validation loss finished higher. This is the start of overfitting. Dropout limits the gap, but 50 epochs already move the model from generalization toward memorization.

Training metrics include active dropout masks. Validation disables dropout. This can make validation accuracy higher than the final training accuracy, as seen for sigmoid and the large network.

The large network was still improving after 20 epochs. With its wide fan-out and learning rate of 0.005, it can benefit from more epochs. Its current validation accuracy does not indicate convergence.

### Prediction and model loading

On the Ryzen system, the complete `predict` command took 741.8 ± 78.3 µs over 5,000 runs. Peak memory was 5.6 MiB. This run used `OPENBLAS_NUM_THREADS=1`.

The measurement includes process startup, image decoding, loading two model files, validating them, running both forward passes, and writing the output. These fixed costs can dominate a single-image run.

The activation choice has little effect on this end-to-end latency at the current model sizes. The sample image is also close to a decision boundary: small differences between training runs can change whether sigmoid, ReLU, or both classify it correctly. One prediction is therefore a functional example, not an accuracy measurement.

Serialization writes the operation plan, arena layout, and `f32` parameters. Each file starts with a magic number and format version. Deserialization checks the operation order, arena ranges, parameter count, and payload length before it allocates the parameter buffer. Sessions are runtime state and are not stored.

Use `perf stat` and `/usr/bin/time -v` on Linux. Use Instruments and `/usr/bin/time -l` on macOS.
