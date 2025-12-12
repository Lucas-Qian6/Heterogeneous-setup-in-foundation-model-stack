
import os
import time
import torch
import torch.distributed as dist
from fms.distributed.strategy import RingAttentionStrategy
from fms.modules.attention import MultiHeadAttention
from fms.utils import print0

# This benchmark compares the performance of Ring Attention under heterogeneous
# GPU setups when using an even split of the sequence vs. a proportional split.

# To simulate a heterogeneous environment, you can use CUDA MPS to limit the
# active thread percentage for some GPUs. For example, to run on 2 GPUs where
# one is twice as fast as the other:
#
# CUDA_VISIBLE_DEVICES=0,1 \
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100,50 \
# torchrun --nproc_per_node=2 hpml_testing/benchmark_het_ring_attn_split.py


def _get_even_split(seq_len, world_size):
    """Splits seq_len as evenly as possible across world_size."""
    base = seq_len // world_size
    rem = seq_len % world_size
    return [base + 1] * rem + [base] * (world_size - rem)


def _get_proportional_split(seq_len, speeds):
    """Splits seq_len proportionally based on relative speeds."""
    world_size = len(speeds)
    total_speed = sum(speeds)
    proportions = [s / total_speed for s in speeds]

    splits = [int(p * seq_len) for p in proportions]

    # Adjust for rounding errors to ensure the sum is correct
    remainder = seq_len - sum(splits)
    for i in range(remainder):
        splits[i] += 1

    # In case of rounding down to zero, give it a single token
    # and steal from the largest. This is a bit of a hack.
    for i in range(world_size):
        if splits[i] == 0:
            splits[i] = 1
            splits[splits.index(max(splits))] -= 1

    return splits


def benchmark_strategy(
    strategy,
    seq_len,
    n_heads,
    d_head,
    n_iters=10,
    n_warmup=5,
):
    """Runs the attention benchmark for a given strategy."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create model and input tensor
    mha = MultiHeadAttention(
        n_heads,
        d_head,
        n_heads * d_head,
        kv_heads=n_heads,
        p_attn=0,
        distributed_strategy=strategy,
    ).to(rank)
    x = torch.randn(1, seq_len, n_heads * d_head).to(rank)

    # Warmup iterations
    for _ in range(n_warmup):
        mha(x)
    dist.barrier()

    # Timed iterations
    start_time = time.time()
    for _ in range(n_iters):
        mha(x)
    dist.barrier()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = total_time / n_iters
    return avg_latency


def main():
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Benchmark parameters
    seq_len = 4096
    n_heads = 32
    d_head = 128

    print0("--------------------------------------------------")
    print0(f"Starting Ring Attention Benchmark")
    print0(f"World Size: {world_size}")
    print0(f"Sequence Length: {seq_len}")
    print0(f"Heads: {n_heads}, Dim/Head: {d_head}")
    print0("--------------------------------------------------")

    # --- Benchmark Even Split ---
    print0("\nRunning benchmark with EVEN split...")
    even_splits = _get_even_split(seq_len, world_size)
    even_strategy = RingAttentionStrategy(block_lens=even_splits)
    print0(f"Even split block_lens: {even_splits}")
    even_latency = benchmark_strategy(even_strategy, seq_len, n_heads, d_head)
    print0(f"Average latency (Even Split): {even_latency:.4f} seconds")

    # --- Benchmark Proportional Split ---
    # Get simulated speeds from env vars (e.g., CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
    # Default to 100% if not set.
    mps_percs_str = os.getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "")
    if mps_percs_str:
        try:
            speeds = [int(p) for p in mps_percs_str.split(",")]
            if len(speeds) != world_size:
                print0(f"Warning: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE has {len(speeds)} entries, but world size is {world_size}. Ignoring.")
                speeds = [100] * world_size
        except ValueError:
            print0(f"Warning: Could not parse CUDA_MPS_ACTIVE_THREAD_PERCENTAGE. Using equal speeds.")
            speeds = [100] * world_size
    else:
        print0("Info: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE not set. Assuming equal speeds.")
        speeds = [100] * world_size

    print0("\nRunning benchmark with PROPORTIONAL split...")
    prop_splits = _get_proportional_split(seq_len, speeds)
    # Proportional split only makes sense if speeds are actually different
    if prop_splits == even_splits:
        print0("Speeds are equal, proportional split is the same as even split. Skipping.")
        prop_latency = even_latency
    else:
        prop_strategy = RingAttentionStrategy(block_lens=prop_splits)
        print0(f"Proportional split block_lens (based on speeds {speeds}): {prop_splits}")
        prop_latency = benchmark_strategy(prop_strategy, seq_len, n_heads, d_head)

    print0(f"Average latency (Proportional Split): {prop_latency:.4f} seconds")


    # --- Results ---
    print0("\n------------------ RESULTS ------------------")
    print0(f"Avg Latency (Even Split):       {even_latency:.4f}s")
    print0(f"Avg Latency (Proportional Split): {prop_latency:.4f}s")
    if prop_latency < even_latency:
        improvement = (even_latency - prop_latency) / even_latency * 100
        print0(f"\nProportional split was {improvement:.2f}% faster.")
    elif even_latency < prop_latency:
        slowdown = (prop_latency - even_latency) / prop_latency * 100
        print0(f"\nEven split was {slowdown:.2f}% faster (this may happen if overheads dominate).")
    else:
        print0("\nBoth strategies performed identically.")
    print0("---------------------------------------------")


if __name__ == "__main__":
    main()
