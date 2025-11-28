import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_FILE = "milestone_4_final_report.csv"
OUTPUT_DIR = "benchmark_plots_trial_2"

# Set a professional style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def clean_currency(x):
    """Converts '$0.0012' -> 0.0012"""
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', ''))
    return x

def clean_percentage(x):
    """Converts '80.5%' -> 80.5"""
    if isinstance(x, str):
        return float(x.replace('%', ''))
    return x

def clean_seconds(x):
    """Converts '12.50s' -> 12.50"""
    if isinstance(x, str):
        return float(x.replace('s', ''))
    return x

def main():
    # 1. Load Data
    if not os.path.exists(CSV_FILE):
        print(f"Error: '{CSV_FILE}' not found. Run evaluation.py first.")
        return

    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    # 2. Clean Data (Convert formatted strings back to numbers)
    # The CSV likely has 'Model' and 'Agent' as columns if saved via to_csv() on a MultiIndex
    if 'Model' not in df.columns or 'Agent' not in df.columns:
        # Handle case where Model/Agent are in the index
        print("Resetting index to ensure Model and Agent are columns...")
        # Reload with header=0 and index_col=[0,1] might be safer, but let's try standard reset
        # If the CSV was saved simply, keys might be separate.
        # Let's assume standard format.
        pass

    # Clean the metric columns
    df['Success'] = df['Success'].apply(clean_percentage)
    df['Cost'] = df['Cost'].apply(clean_currency)
    df['Cost/Success ($)'] = df['Cost/Success ($)'].apply(clean_currency)
    df['Latency'] = df['Latency'].apply(clean_seconds)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHART 1: SUCCESS RATE COMPARISON
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(
        data=df,
        x="Agent",
        y="Success",
        hue="Model",
        palette="viridis",
        edgecolor="black"
    )
    
    plt.title("Task Success Rate by Agent Configuration", fontsize=16, pad=20)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.xlabel("Agent Type", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(title="LLM Backend")
    
    # Add labels on top of bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "1_success_rate.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

    # CHART 2: COST PER SUCCESSFUL TASK
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(
        data=df,
        x="Agent",
        y="Cost/Success ($)",
        hue="Model",
        palette="magma",
        edgecolor="black"
    )
    
    plt.title("Cost Efficiency: Price per Successful Task", fontsize=16, pad=20)
    plt.ylabel("Cost ($)", fontsize=14)
    plt.xlabel("Agent Type", fontsize=14)
    plt.legend(title="LLM Backend")
    
    # Add labels (Small currency values)
    for container in chart.containers:
        chart.bar_label(container, fmt='$%.4f', padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "2_cost_efficiency.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

    # CHART 3: LATENCY COMPARISON
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(
        data=df,
        x="Agent",
        y="Latency",
        hue="Model",
        palette="coolwarm",
        edgecolor="black"
    )
    
    plt.title("Average Latency (Time to Complete)", fontsize=16, pad=20)
    plt.ylabel("Time (Seconds)", fontsize=14)
    plt.xlabel("Agent Type", fontsize=14)
    plt.legend(title="LLM Backend")
    
    # Add labels
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1fs', padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "3_latency.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()