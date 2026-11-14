import pandas as pd
import numpy as np
import os
from scipy import stats

def analyze_ml_results():
    input_path = "results/ml_experiment/raw_results.csv"
    output_dir = "results/ml_experiment/analysis"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Total dataset records: {len(df)}")
    assert len(df) == 900, "Experiment incomplete: Expected 900 records (6 algos * 5 benchmarks * 30 seeds)."

    # 1. Statistical Summary
    stats_df = df.groupby(['Benchmark', 'Algorithm'])['BestFitness'].agg(
        Mean='mean', Median='median', Std='std', Min='min', Max='max'
    ).reset_index()
    stats_df.to_csv(f"{output_dir}/statistical_summary.csv", index=False)

    # 2. Friedman Ranks
    df['rank'] = df.groupby(['Benchmark', 'Seed'])['BestFitness'].rank(method='average')
    rank_overall = df.groupby('Algorithm')['rank'].mean().reset_index().rename(columns={'rank': 'Overall_Rank'})
    rank_per_b = df.groupby(['Benchmark', 'Algorithm'])['rank'].mean().unstack().T.reset_index()

    rank_summary = pd.merge(rank_overall, rank_per_b, on='Algorithm').sort_values('Overall_Rank')
    rank_summary.to_csv(f"{output_dir}/friedman_ranks.csv", index=False)

    # 3. Wilcoxon Signed-Rank Test vs Base HAOA (HAOA_orig)
    wilcoxon_results = []
    for b in df['Benchmark'].unique():
        for algo in ['HAOA_impr', 'HAOA_adv', 'MLHAOA', 'PSO', 'DE']:
            data_orig = df[(df['Benchmark']==b) & (df['Algorithm']=='HAOA_orig')]['BestFitness'].values
            data_test = df[(df['Benchmark']==b) & (df['Algorithm']==algo)]['BestFitness'].values
            stat, p = stats.wilcoxon(data_orig, data_test, alternative='two-sided')
            wilcoxon_results.append({
                'Benchmark': b,
                'Comparison': f"{algo} vs HAOA_orig",
                'Statistic': stat,
                'p_value': p,
                'Significant_p05': p < 0.05
            })
    pd.DataFrame(wilcoxon_results).to_csv(f"{output_dir}/wilcoxon_tests.csv", index=False)

    print("\n--- Summary Exported Successfully ---")
    print("\nOverall Ranks:")
    print(rank_summary[['Algorithm', 'Overall_Rank']].to_string(index=False))

if __name__ == "__main__":
    analyze_ml_results()
