# About 

This repository contains alpha stage (highly WiP) primitives for typical operations required for various pairing based proof systems - finite field operations, FFT, EC point arithmetic and multiexponentiation. All operations implemented are only for G1 group of pairing friendly curve BN254 (Ethereum curve).

There is also an implementation of 70% of the work required for Groth16 prover with interface for intergration with `bellman_ce` Rust crate.

## Instruction

- Install CUDA 10.0 and Cmake 3.9+
- `cmake --release .`
- `make`
- in some folder clone `ff`, `pairing` and `bellman` repositories from Matter Labs
- checkout `gpu` branches in all of them
- copy file `sources/libcuda.so` into the `bellman` directory
- try to run tests `cargo test --release -- --nocapture test_mimc_bn256_gpu_all` to get some benchmark results and validity checks.
- you can also change `const MIMC_ROUNDS: usize = 16000000;` in a file `bellman/tests/mimc.rs` to reduce number of constrains in a test circuits if you run out of memory (RAM or GPU memory)