# Ring Attention Implementation

## Modified Files
- `fms/distributed/strategy.py`
    Added `RingAttentionStrategy` class with separate CUDA streams for communication and compute overlap. Added `block_lens` parameter for uneven token distribution across GPUs (still a work in progress). 
- `fms/models/__init__.py`
    Added `"ring"` as a distributed strategy option, parses `block_lens` from kwargs.
- `fms/models/llama.py`
    Modified `LLaMABlock` and `LLaMAHeadless` to support ring attention. 

## New Files
- `fms/distributed/ring_attention.py`
    This file contains the core ring attention implementation. The `ring_forward()` method is called from `LLaMABlock` and replaces standard attention. `_compute_attention_ring_pass_kv()` implements the main ring loop. It uses two CUDA streams (default for compute and a dedicated comm stream) so KV communication can overlap with compute. For online softmax merging, it computes per-block softmax statistics; a custom Triton kernel is used **conditionally** (only when the block workload is large enough), and otherwise it falls back to a correct PyTorch implementation.
- `fms/distributed/triton_block.py`
    Custom Triton kernel that computes per-block softmax statistics used in online softmax merging. In the current implementation it is invoked only when the block size/work is above a threshold; smaller blocks use a PyTorch fallback to avoid Triton launch overhead.
- `hpml_testing/`
    Contains many files for testing and benchmarking. 

## Current Progress
- Implemented Triton kernel for ring attention with online softmax. Benchmarked on heterogeneous GPU 
  simulation. Currently working on logic for passing in uneven Q lengths across GPUs.

