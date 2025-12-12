import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
DEFAULT_INPUT_CSV = "hpml_testing/results/sweep_results.csv"
OUTPUT_DIR = "hpml_testing/plots"

def create_plots(csv_path):
    """
    Reads the sweep results from a CSV file and generates plots
    to compare the performance of different split strategies.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- 1. Grouped Bar Charts ---
    # This plot directly compares the latency of the three methods for each scenario.
    print("Generating grouped bar charts...")
    g_bar = sns.catplot(
        data=df,
        x="slowdown_pct",
        y="overall_latency_ms",
        hue="split_type",
        col="seq_len",
        kind="bar",
        height=5,
        aspect=1.2,
        sharey=False,  # Allow y-axes to have different scales for each facet
        legend_out=True
    )
    g_bar.fig.suptitle("Latency Comparison by Sequence Length and Slowdown", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Overall Latency (ms)")
    g_bar.set_titles("Sequence Length: {col_name}")
    g_bar.despine(left=True)
    for ax in g_bar.axes.flat:
        ax.set_yscale('log') # Use log scale as latency can vary greatly
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_chart_path = os.path.join(OUTPUT_DIR, "sweep_latency_bar_charts.png")
    g_bar.savefig(bar_chart_path)
    print(f"Bar charts saved to {bar_chart_path}")


    # --- 2. Line Charts ---
    # This plot shows the performance trend and robustness of each method.
    print("Generating line charts...")
    g_line = sns.relplot(
        data=df,
        x="slowdown_pct",
        y="overall_latency_ms",
        hue="split_type",
        style="split_type",
        col="seq_len",
        kind="line",
        height=5,
        aspect=1.2,
        sharey=False,
        legend="full",
        markers=True,
        dashes=False
    )
    g_line.fig.suptitle("Latency Trend by Sequence Length and Slowdown", y=1.03)
    g_line.set_axis_labels("Slowdown Percentage for Rank 1", "Overall Latency (ms)")
    g_line.set_titles("Sequence Length: {col_name}")
    g_line.set(xdir="reverse") # Show slowdown getting worse from left to right
    g_line.despine(left=True)
    for ax in g_line.axes.flat:
        ax.set_yscale('log') # Use log scale
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    line_chart_path = os.path.join(OUTPUT_DIR, "sweep_latency_line_charts.png")
    g_line.savefig(line_chart_path)
    print(f"Line charts saved to {line_chart_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV
    
    create_plots(input_csv)
