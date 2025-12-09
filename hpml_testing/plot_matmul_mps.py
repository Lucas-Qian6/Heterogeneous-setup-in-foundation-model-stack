"""
Plot matmul MPS sweep results (log2 x-axis, linear y-axis).

Expects a CSV with columns:
  mps_pct,size,dtype,bg_streams,avg_ms,tflops

Usage:
  python hpml_testing/plot_matmul_mps.py hpml_testing/results/matmul_mps_sweep_*.csv
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_matmul_mps.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    for mps_pct, group in df.groupby("mps_pct"):
        group = group.sort_values("size")
        plt.plot(
            group["size"],
            group["avg_ms"],
            marker="o",
            label=f"MPS {mps_pct}%",
        )

    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Average latency (ms)")
    plt.title("Matmul latency vs size (log2 x-axis)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xscale("log", base=2)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
