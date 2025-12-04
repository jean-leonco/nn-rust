# nn-rust

nn-rust is a toy neural network library written in Rust. This project was built for educational purposes to understand the internals of deep learning, backpropagation, and matrix operations.

## Features

- Built from Scratch: Understanding the core logic behind neural networks without high-level frameworks.
- Modular Architecture: A chainable builder pattern to construct custom models.
- Layer Support:
  - Dense (Fully Connected) Layers
  - Activation Functions: ReLU, Sigmoid
  - Loss Layer: Softmax Cross Entropy
- Hardware Acceleration: Leverages `ndarray` with BLAS (Accelerate on macOS, OpenBLAS on Linux/Windows).
- MNIST Integration: Includes a specialized loader for the MNIST handwritten digit dataset.

## Prerequisites

- Rust: Latest stable version.
- BLAS Backend:
  - macOS: Uses the built-in `accelerate` framework automatically.
  - Linux: Requires [OpenBLAS](http://www.openmathlib.org/OpenBLAS/docs/install/).

## Quick Start

1.  Clone the repo:

    ```sh
    git clone [https://github.com/jean-leonco/nn-rust.git](https://github.com/jean-leonco/nn-rust.git)
    cd nn-rust
    ```

2.  Build: Build the project in release mode for best performance.

    ```sh
    cargo build --release
    ```

3.  Train: Run the training binary to train a ReLU and a Sigmoid model on MNIST.

    ```sh
    ./target/release/train
    ```

4.  Predict: Run the models and predict on a test image:
    ```sh
    ./target/release/predict
    ```

## Code Structure

- `src/bin/`: Example executables (`train.rs` and `predict.rs`) demonstrating how to use the library.
- `src/dataloader/`: Contains the MNIST `DataLoader` struct implementation.
- `src/layer/`: Implementations of Dense, ReLU, and Sigmoid layers.
- `src/model/`: Contains the `Model` struct and `Builder` implementation.

## Benchmarks

Comparing performance between Linux (OpenBLAS) and macOS (Accelerate). The macOS build leverages the Apple Accelerate framework, resulting in significantly faster training times for matrix-heavy operations.

### System Specs

- System A (Linux): AMD Ryzen 5 5600X | Linux 6.17.10
- System B (macOS): Apple M3 Pro (5P + 6E Cores) | macOS

### Training Performance (15 Epochs)

| System            | Wall Clock Time | Peak Memory |
| :---------------- | :-------------- | :---------- |
| AMD Ryzen 5 5600X | 20.96s          | ~282 MB     |
| Apple M3 Pro      | 4.36s           | ~318 MB     |

Final Model Stats (M3 Pro Run):

| Model   | Final Train Loss | Final Train Acc | Validation Loss | Validation Acc |
| :------ | :--------------- | :-------------- | :-------------- | :------------- |
| ReLU    | 0.0514           | 98.59%          | 0.0803          | 97.49%         |
| Sigmoid | 0.1625           | 95.32%          | 0.1670          | 94.91%         |

### Inference Performance

| System            | Wall Clock Time | Peak Memory |
| :---------------- | :-------------- | :---------- |
| AMD Ryzen 5 5600X | < 0.01s         | ~8 MB       |
| Apple M3 Pro      | < 0.01s         | ~8 MB       |

### Inference Performance

- Wall Clock Time: < 0.01s (Instant)
- Peak Memory Usage: ~8 MB (8,320 kbytes)

| Model   | Predicted | Actual | Result                |
| :------ | :-------- | :----- | :-------------------- |
| ReLU    | 3         | 3      | Correct (76.0% conf)  |
| Sigmoid | 3         | 3      | Correct (83.26% conf) |
