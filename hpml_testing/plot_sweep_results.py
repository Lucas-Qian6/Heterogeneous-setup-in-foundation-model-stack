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
    to show the efficiency of different strategies relative to the ideal homogeneous baseline.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at {csv_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- Data Transformation for Efficiency Calculation ---
    # Pivot the table to get latency for each split_type in columns
    pivot_df = df.pivot_table(
        index=['seq_len', 'slowdown_pct'],
        columns='split_type',
        values='overall_latency_ms'
    ).reset_index()

    # Calculate efficiency relative to the 'reference_homogeneous' baseline
    # Efficiency = Time_ideal / Time_measured
    pivot_df['even_efficiency'] = pivot_df['reference_homogeneous'] / pivot_df['even']
    pivot_df['uneven_efficiency'] = pivot_df['reference_homogeneous'] / pivot_df['uneven']
    pivot_df['lut_efficiency'] = pivot_df['reference_homogeneous'] / pivot_df['lut']

    # Melt the DataFrame back into a long format for plotting
    efficiency_df = pivot_df.melt(
        id_vars=['seq_len', 'slowdown_pct'],
        value_vars=['even_efficiency', 'uneven_efficiency', 'lut_efficiency'],
        var_name='split_type',
        value_name='efficiency'
    )
    # Clean up the 'split_type' names
    efficiency_df['split_type'] = efficiency_df['split_type'].str.replace('_efficiency', '')


    # --- Grouped Bar Charts for Efficiency ---
    print("Generating efficiency bar charts...")
    g_bar = sns.catplot(
        data=efficiency_df,
        x="slowdown_pct",
        y="efficiency",
        hue="split_type",
        col="seq_len",
        kind="bar",
        height=5,
        aspect=1.2,
        sharey=True,
        legend_out=True
    )
    g_bar.fig.suptitle("Heterogeneous Performance Efficiency vs Ideal Homogeneous Case", y=1.03)
    g_bar.set_axis_labels("Slowdown Percentage for Rank 1", "Efficiency (Higher is Better)")
    g_bar.set_titles("Sequence Length: {col_name}")
    g_bar.despine(left=True)

    # Set y-axis limit and add a horizontal line at y=1.0 to represent the ideal performance
    for ax in g_bar.axes.flat:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal Performance')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Handle legend
    plt.legend()
    
    bar_chart_path = os.path.join(OUTPUT_DIR, "sweep_efficiency_bar_charts.png")
    g_bar.savefig(bar_chart_path)
    print(f"Efficiency bar charts saved to {bar_chart_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT_CSV
    
    create_plots(input_csv)
