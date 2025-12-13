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
    Reads the sweep results from a CSV file and generates a grouped bar chart
    to compare the raw overall latency of the four different strategies.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- Grouped Bar Charts for Raw Latency ---
    print("Generating latency comparison bar charts...")
    g_bar = sns.catplot(
        data=df,
        x="slowdown_pct",
        y="overall_latency_ms",
        hue="split_type",
        col="seq_len",
        kind="bar",
        height=6,
        aspect=1.3,
        sharey=False,  # Latency values will vary greatly across facets
        legend_out=True
    )
    g_bar.fig.suptitle("Latency Comparison of Rebalancing Strategies", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Overall Latency (ms) [Log Scale]")
    g_bar.set_titles("Sequence Length: {col_name}")
    g_bar.despine(left=True)

    # Use a log scale for the y-axis to visualize large differences
    for ax in g_bar.axes.flat:
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_chart_path = os.path.join(OUTPUT_DIR, "sweep_latency_comparison.png")
    g_bar.savefig(bar_chart_path)
    print(f"Latency comparison bar charts saved to {bar_chart_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV
    
    create_plots(input_csv)
