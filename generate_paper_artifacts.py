import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# --- Configuration & Palette (DataViz compliant) ---
OUTPUT_FIGURES = "results/ml_experiment/figures"
OUTPUT_TABLES = "results/ml_experiment/analysis"
os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

# Dataviz colorblind-safe palette (OKLab validated contrast, distinct hues)
# ML-HAOA: Deep Violet (#6b21a8), HAOA_adv: Indigo (#2563eb), HAOA_impr: Cyan (#0891b2), HAOA_orig: Amber (#d97706), PSO: Emerald (#059669), DE: Rose (#e11d48)
COLORS = {
    "MLHAOA": "#6b21a8",     # Primary (Ours) - Deep Violet
    "HAOA_adv": "#2563eb",   # Indigo
    "HAOA_impr": "#0891b2",  # Cyan
    "HAOA_orig": "#d97706",  # Amber
    "PSO": "#059669",        # Emerald
    "DE": "#e11d48"          # Rose
}

LABELS = {
    "MLHAOA": "ML-HAOA (Ours)",
    "HAOA_adv": "HAOA-Advanced",
    "HAOA_impr": "HAOA-Improved",
    "HAOA_orig": "HAOA-Original",
    "PSO": "PSO",
    "DE": "DE"
}

STYLES = {
    "MLHAOA": {"lw": 2.5, "ls": "-"},
    "HAOA_adv": {"lw": 1.8, "ls": "--"},
    "HAOA_impr": {"lw": 1.8, "ls": "-."},
    "HAOA_orig": {"lw": 1.8, "ls": ":"},
    "PSO": {"lw": 1.5, "ls": "--"},
    "DE": {"lw": 1.5, "ls": "-."}
}

BENCHMARKS = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"]
ALGORITHMS = ["MLHAOA", "HAOA_adv", "HAOA_impr", "HAOA_orig", "PSO", "DE"]


def load_convergence_histories():
    histories = {b: {a: [] for a in ALGORITHMS} for b in BENCHMARKS}
    conv_dir = "results/ml_experiment/convergence_histories"

    for b in BENCHMARKS:
        for a in ALGORITHMS:
            files = sorted(glob.glob(os.path.join(conv_dir, f"{a}_{b}_*.csv")))
            # filter out diversity files
            files = [f for f in files if "_diversity" not in f]
            for f in files:
                try:
                    series = pd.read_csv(f).squeeze("columns").values
                    histories[b][a].append(series)
                except Exception:
                    pass
    return histories


def generate_convergence_plots(histories):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, b in enumerate(BENCHMARKS):
        ax = axes[idx]
        for a in ALGORITHMS:
            runs = histories[b][a]
            if not runs:
                continue
            # Trim runs to minimum length
            min_len = min(len(r) for r in runs)
            matrix = np.array([r[:min_len] for r in runs])

            mean_curve = np.mean(matrix, axis=0)
            std_curve = np.std(matrix, axis=0)
            x_vals = np.arange(1, min_len + 1)

            ax.plot(x_vals, mean_curve, color=COLORS[a], label=LABELS[a],
                    linewidth=STYLES[a]["lw"], linestyle=STYLES[a]["ls"])
            # Shaded 95% confidence interval or standard error band
            sem_curve = std_curve / np.sqrt(len(runs))
            ax.fill_between(x_vals, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=COLORS[a], alpha=0.12)

        ax.set_title(f"({chr(97+idx)}) {b} Benchmark", fontweight="bold", loc="left")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Best Score (Fitness)")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Put legend in 6th subplot cell
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', borderpad=1.5, fontsize=11)
    axes[5].set_title("Algorithm Legend", fontweight="bold", loc="center")

    plt.suptitle("Convergence Performance Across 5 Standard Benchmarks (30 Independent Seeds)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUTPUT_FIGURES, "fig1_convergence_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig("HAOA/results/benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot: {out_path}")


def generate_boxplot_comparison(raw_df):
    plt.figure(figsize=(14, 7))
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    for idx, b in enumerate(BENCHMARKS):
        ax = axes[idx]
        b_data = raw_df[raw_df["Benchmark"] == b]

        data_to_plot = [b_data[b_data["Algorithm"] == a]["BestFitness"].values for a in ALGORITHMS]

        bp = ax.boxplot(data_to_plot, labels=[LABELS[a].replace(" ", "\n") for a in ALGORITHMS],
                        patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

        for patch, algo in zip(bp['boxes'], ALGORITHMS):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.7)

        ax.set_title(f"{b}", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle("Distribution of Final Fitness Across 30 Independent Seeds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FIGURES, "fig2_fitness_boxplots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot figure: {out_path}")


def generate_paper_tables(raw_df):
    # Calculate Mean ± Std, Median, Min, Max, Wilcoxon p-values vs HAOA_orig
    table_rows = []

    for b in BENCHMARKS:
        b_df = raw_df[raw_df["Benchmark"] == b]
        orig_vals = b_df[b_df["Algorithm"] == "HAOA_orig"]["BestFitness"].values

        for a in ALGORITHMS:
            a_vals = b_df[b_df["Algorithm"] == a]["BestFitness"].values
            mean_val = np.mean(a_vals)
            std_val = np.std(a_vals)
            med_val = np.median(a_vals)
            min_val = np.min(a_vals)
            max_val = np.max(a_vals)

            if a == "HAOA_orig":
                p_str = "Baseline"
                sig = ""
            else:
                _, p = stats.wilcoxon(orig_vals, a_vals, alternative="two-sided")
                p_str = f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
                sig = "$^{\\dagger}$" if p < 0.05 else ""

            table_rows.append({
                "Benchmark": b,
                "Algorithm": LABELS[a],
                "Mean ± Std": f"{mean_val:.4e} ± {std_val:.4e}",
                "Median": f"{med_val:.4e}",
                "Min": f"{min_val:.4e}",
                "Max": f"{max_val:.4e}",
                "p-value (vs Orig)": f"{p_str}{sig}"
            })

    table_df = pd.DataFrame(table_rows)

    # Save Markdown table
    md_path = os.path.join(OUTPUT_TABLES, "paper_tables.md")
    with open(md_path, "w") as f:
        f.write("# Paper Benchmark Results (NFE=15,030, D=30, 30 Seeds)\n\n")
        f.write("$^{\\dagger}$ indicates statistically significant difference ($p < 0.05$) via two-sided Wilcoxon signed-rank test against HAOA-Original.\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved Markdown paper table: {md_path}")

    # Save LaTeX table
    tex_path = os.path.join(OUTPUT_TABLES, "paper_tables.tex")
    with open(tex_path, "w") as f:
        f.write("% LaTeX Table for Publication Submission\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Performance comparison of ML-HAOA against baselines across 5 benchmark landscapes ($D=30$, $NFE=15,030$, 30 seeds). \\dagger denotes $p < 0.05$ vs HAOA-Original.}\n")
        f.write("\\label{tab:benchmark_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Benchmark & Algorithm & Mean $\\pm$ Std & Median & Min & Max & $p$-value \\\\\n\\hline\n")

        current_b = ""
        for _, row in table_df.iterrows():
            b_label = row['Benchmark'] if row['Benchmark'] != current_b else ""
            current_b = row['Benchmark']
            mean_std = row['Mean ± Std'].replace("e-0", "\\times 10^{-").replace("e+", "\\times 10^{").replace(" ± ", "} \\pm ")
            f.write(f"{b_label} & {row['Algorithm']} & {mean_std} & {row['Median']} & {row['Min']} & {row['Max']} & {row['p-value (vs Orig)']} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved LaTeX paper table: {tex_path}")


if __name__ == "__main__":
    raw_df = pd.read_csv("results/ml_experiment/raw_results.csv")
    histories = load_convergence_histories()

    generate_convergence_plots(histories)
    generate_boxplot_comparison(raw_df)
    generate_paper_tables(raw_df)
