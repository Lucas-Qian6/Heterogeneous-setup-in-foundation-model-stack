import os
import subprocess
import re
import pandas as pd
import math
import time
import argparse
import wandb

# --- SIMPLE SWEEP for WANDB DEMO ---
SEQLEN_SWEEP = [8192]
SLOWDOWN_SWEEP = [80, 30] 
DEFAULT_PROFILE_PATH = "hpml_testing/results/ring_attention_profile.csv"
BENCHMARK_SCRIPT = "hpml_testing/benchmark_hetero_latency.py"

def parse_single_benchmark_output(output_string):
    results = {}
    split_type_match = re.search(r"Running benchmark with '(\w+)' split.", output_string)
    if split_type_match:
        results['split_type'] = split_type_match.group(1)
    else:
        results['split_type'] = "unknown" 
    seq_len_match = re.search(r"Sequence Length: (\d+), Block lengths: [^]]*]", output_string)
    if seq_len_match:
        results['seq_len'] = int(seq_len_match.group(1))
    overall_latency_match = re.search(r"Overall Latency \(max of ranks\): (\d+\.\d+) ms", output_string)
    if overall_latency_match:
        results['overall_latency_ms'] = float(overall_latency_match.group(1))
    rank_latency_token_matches = re.findall(r"Rank (\d+) \((\d+) tokens\): (\d+\.\d+) ms", output_string)
    for match in rank_latency_token_matches:
        rank = int(match[0])
        tokens = int(match[1])
        latency = float(match[2])
        results[f'rank{rank}_tokens'] = tokens
        results[f'rank{rank}_latency_ms'] = latency
    return results

def main():
    parser = argparse.ArgumentParser(description="Run a simple sweep and log results to W&B.")
    parser.add_argument(
        "--profile-path",
        type=str,
        default=DEFAULT_PROFILE_PATH,
        help="Path to the performance profile (LUT) to use for the 'lut' strategy."
    )
    args = parser.parse_args()

    wandb.init(
        project="heterogeneous-ring-attention-demo",
        config={
            "seq_len_sweep": SEQLEN_SWEEP,
            "slowdown_sweep": SLOWDOWN_SWEEP,
            "profile_path": args.profile_path
        }
    )
    print("W&B Run Initialized. Starting fast benchmark sweep for demo...")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    all_sweep_results = []
    os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

    try:
        for seq_len in SEQLEN_SWEEP:
            for slowdown_pct in SLOWDOWN_SWEEP:
                print(f"\n--- Running: SEQ_LEN={seq_len}, SLOWDOWN_PCT={slowdown_pct} ---")
                current_slowdown_factor = slowdown_pct / 100.0

                strategies_to_run = ["even", "uneven", "lut", "regression", "reference_homogeneous"]
                for split_type in strategies_to_run:
                    print(f"Running {split_type} split...")
                    
                    base_cmd = ["python3", BENCHMARK_SCRIPT, "--rank", "0", "--world-size", "2", "--seq-len", str(seq_len), "--emb-dim", "4096", "--n-heads", "32"]
                    
                    if split_type == "reference_homogeneous":
                        cmd_args = ["--split-type", "even"]
                    else:
                        cmd_args = ["--split-type", split_type]

                    if split_type == "uneven":
                        cmd_args.extend(["--slowdown-factor", str(current_slowdown_factor)])
                    if split_type in ["lut", "formula"]:
                         cmd_args.extend(["--use-perf-profile", args.profile_path, "--rank-mps", f"100,{slowdown_pct}"])

                    p0_cmd = base_cmd + cmd_args
                    p1_cmd = base_cmd.copy()
                    p1_cmd[4] = "1"
                    p1_cmd.extend(cmd_args)
                    
                    p0 = subprocess.Popen(p0_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    p1_env = os.environ.copy()
                    if split_type != "reference_homogeneous":
                        p1_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(slowdown_pct)
                        
                    p1 = subprocess.Popen(p1_cmd, env=p1_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    stdout_rank0, stderr_rank0 = p0.communicate()
                    _, stderr_rank1 = p1.communicate()

                    if p0.returncode != 0:
                        print(f"Error in {split_type} split (rank 0): {stderr_rank0}")
                    if p1.returncode != 0:
                        print(f"Error in {split_type} split (rank 1): {stderr_rank1}")
                    
                    results = parse_single_benchmark_output(stdout_rank0)
                    results.update({
                        "seq_len": seq_len,
                        "slowdown_pct": slowdown_pct,
                        "split_type": split_type,
                    })
                    all_sweep_results.append(results)

                time.sleep(1)
                
    except Exception as e:
        print(f"An error occurred during the demo sweep: {e}")
    finally:
        print("\nFast sweep finished. Logging results to W&B...")
        if all_sweep_results:
            df_results = pd.DataFrame(all_sweep_results)
            # Log the DataFrame as a W&B Table
            wandb.log({"demo_results": wandb.Table(dataframe=df_results)})
            print("Results logged to W&B.")
        else:
            print("No results were generated.")
        
        wandb.finish()
        print("W&B run finished.")

if __name__ == "__main__":
    main()
