import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set Seaborn style
sns.set_style('whitegrid')

def adaptive_few_shot_sampling(df, country_config):
    """Adaptive range-based sampling"""
    
    # Define priority ranges (in order of clinical importance)
    priority_bins = [
        (3, 7),   # Very important: transfusion threshold
        (7, 10),  # Important: anemia range
        (13, 16), # Important: high values
        (10, 13)  # Normal: normal range
    ]
    
    target = 10 if country_config != 'joint' else 20
    selected = []
    
    # Step 1: Secure available samples from priority ranges
    for bin_range in priority_bins:
        bin_data = df[(df['Hb'] >= bin_range[0]) & (df['Hb'] < bin_range[1])]
        
        if len(bin_data) > 0:
            # Determine sample number proportional to range size
            max_from_bin = min(3, len(bin_data))  # Maximum 3 per range
            n_samples = min(max_from_bin, target - len(selected))
            
            if n_samples > 0:
                sampled = bin_data.sample(n=n_samples, random_state=42)
                selected.extend(sampled.index.tolist())
        
        if len(selected) >= target:
            break
    
    # Step 2: Fill shortage with weighted random sampling
    if len(selected) < target:
        remaining = target - len(selected)
        available = df.index.difference(selected)
        
        if len(available) >= remaining:
            # Weighted sampling considering Hb distribution
            weights = calculate_clinical_weights(df.loc[available]['Hb'])
            additional = df.loc[available].sample(
                n=remaining, 
                weights=weights,
                random_state=42
            )
            selected.extend(additional.index.tolist())
    
    return selected[:target]

def calculate_clinical_weights(hb_values):
    """Clinical importance-based weights"""
    weights = []
    for hb in hb_values:
        if hb < 7:
            weight = 3.0    # Very important
        elif hb < 10:
            weight = 2.0    # Important
        elif hb > 15:
            weight = 2.0    # Important
        else:
            weight = 1.0    # Default
        weights.append(weight)
    
    return weights

# Strategy definition
india_strategy = {
    'target_distribution': {
        (7, 10): 3,   # Secure anemia range
        (10, 13): 4,  # Leverage existing strengths
        (13, 16): 3   # Supplement high values
    }
}

italy_strategy = {
    'target_distribution': {
        (7, 10): 4,   # Maximize low values
        (10, 13): 3,  # Supplement middle values
        (13, 16): 3   # Utilize existing data
    }
}

joint_strategy = {
    'per_country': {
        'India': italy_strategy,  # Mutual complement
        'Italy': india_strategy
    }
}

def calculate_fold_metrics(internal_df):
    """Calculate MAE, STD per fold"""
    fold_metrics = {}
    for fold in range(5):  # 0-4 fold
        fold_data = internal_df[internal_df['fold'] == fold]
        if len(fold_data) > 0:
            mae = np.mean(np.abs(fold_data['ground truth'] - fold_data['prediction']))
            std = np.std(np.abs(fold_data['ground truth'] - fold_data['prediction']))
            fold_metrics[fold] = {'mae': mae, 'std': std}
    return fold_metrics

def select_best_checkpoint(fold_metrics):
    """Select best fold using composite score"""
    if not fold_metrics:
        return None, None
    
    # Collect values from all folds
    maes = [fold_metrics[fold]['mae'] for fold in fold_metrics]
    stds = [fold_metrics[fold]['std'] for fold in fold_metrics]
    cvs = [fold_metrics[fold]['std'] / fold_metrics[fold]['mae'] for fold in fold_metrics]
    
    # Calculate min/max for normalization
    mae_min, mae_max = min(maes), max(maes)
    std_max = max(stds)
    cv_max = max(cvs)
    
    scores = {}
    for fold in fold_metrics:
        mae = fold_metrics[fold]['mae']
        std = fold_metrics[fold]['std']
        cv = std / mae
        
        # Normalize
        mae_norm = (mae - mae_min) / (mae_max - mae_min) if mae_max != mae_min else 0
        std_norm = std / std_max if std_max != 0 else 0
        cv_norm = cv / cv_max if cv_max != 0 else 0
        
        # Weights optimized for external validation
        score = 0.5 * mae_norm + 0.3 * std_norm + 0.2 * cv_norm
        scores[fold] = score
    
    best_fold = min(scores, key=scores.get)
    return best_fold, scores

def analyze_hb_frequency(hb_values, label, hb_ranges):
    """Hb frequency analysis - clear range definition"""
    freq = {}
    total = len(hb_values)
    
    for i, (lower, upper, label_range) in enumerate(hb_ranges):
        if lower is None:  # <6
            count = (hb_values < upper).sum()
        elif upper is None:  # >=15
            count = (hb_values >= lower).sum()
        else:  # Clear range: lower <= Hb < upper
            count = ((hb_values >= lower) & (hb_values < upper)).sum()
        
        freq[label_range] = {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }
    
    return freq

def main():
    # Define Hb ranges - clear intervals: (lower, upper, label)
    # lower <= Hb < upper (both boundaries clear)
    hb_ranges = [
        (None, 6, "<6"),              # Hb < 6
        (6, 7, "6-6.9"),              # 6 <= Hb < 7
        (7, 8, "7-7.9"),              # 7 <= Hb < 8
        (8, 9, "8-8.9"),              # 8 <= Hb < 9
        (9, 10, "9-9.9"),             # 9 <= Hb < 10
        (10, 11, "10-10.9"),          # 10 <= Hb < 11
        (11, 12, "11-11.9"),          # 11 <= Hb < 12
        (12, 13, "12-12.9"),          # 12 <= Hb < 13
        (13, 14, "13-13.9"),          # 13 <= Hb < 14
        (14, 15, "14-14.9"),          # 14 <= Hb < 15
        (15, None, ">=15")            # Hb >= 15
    ]
    
    # Load Internal data
    internal_df = pd.read_excel('results_excel_file_final/results_image-bins76_combined.xlsx')
    internal_hb = internal_df['ground truth']
    
    # Load External data
    external_master_df = pd.read_csv('external_validation_joint_results/external_validation_master.csv')
    external_india_df = pd.read_csv('external_validation_joint_results/external_validation_india.csv')
    external_italy_df = pd.read_csv('external_validation_joint_results/external_validation_italy.csv')
    # external_ghana_df = pd.read_excel('external_validation_ghana/anemiaDataGhana.xlsx')
    # print(f"Ghana data columns: {external_ghana_df.columns.tolist()}")
    # print(f"Ghana data sample:\n{external_ghana_df.head()}")
    
    # Process Hb column: exclude empty strings and "_"
    external_master_df = external_master_df[external_master_df['Hb'] != ""]
    external_master_df = external_master_df[external_master_df['Hb'] != "_"]
    external_india_df = external_india_df[external_india_df['Hb'] != ""]
    external_india_df = external_india_df[external_india_df['Hb'] != "_"]
    external_italy_df = external_italy_df[external_italy_df['Hb'] != ""]
    external_italy_df = external_italy_df[external_italy_df['Hb'] != "_"]
    # Process Ghana data (not used)
    # if 'Hb' in external_ghana_df.columns:
    #     external_ghana_df = external_ghana_df[external_ghana_df['Hb'] != ""]
    #     external_ghana_df = external_ghana_df[external_ghana_df['Hb'] != "_"]
    
    external_master_hb = pd.to_numeric(external_master_df['Hb'], errors='coerce').dropna()
    external_india_hb = pd.to_numeric(external_india_df['Hb'], errors='coerce').dropna()
    external_italy_hb = pd.to_numeric(external_italy_df['Hb'], errors='coerce').dropna()
    # external_ghana_hb = pd.to_numeric(external_ghana_df['Hb'], errors='coerce').dropna() if 'Hb' in external_ghana_df.columns else pd.Series(dtype=float)
    
    # Analyze low Hb cases (<8 g/dL)
    print("=== Low Hb Case Analysis (<8 g/dL) ===")
    datasets = {
        'Internal': internal_hb,
        'External (Master)': external_master_hb,
        'External (India)': external_india_hb,
        'External (Italy)': external_italy_hb,
        # 'External (Ghana)': external_ghana_hb
    }
    dataset_sizes = {name: len(hb) for name, hb in datasets.items()}
    highlight_datasets = {'Internal', 'External (Master)'}
    color_map = {
        'Internal': '#1f77b4',          # blue (highlighted)
        'External (Master)': '#ff7f0e', # orange (highlighted)
        'External (India)': '#aaaaaa',  # medium-light gray (soft)
        'External (Italy)': '#666666',  # dark gray (bold and visible)
    # 'External (Ghana)': '#2ca02c'   # green
    }
    
    for name, hb in datasets.items():
        low_count = (hb < 8).sum()
        total = len(hb)
        print(f"{name}: {low_count}/{total} ({low_count/total*100:.1f}%)")
    
    # Evaluate stratification feasibility
    print("\n=== Stratification Feasibility Evaluation ===")
    for name, hb in datasets.items():
        low_count = (hb < 8).sum()
        if low_count >= 5:  # Stratification possible if at least 5 cases
            print(f"{name}: Stratification possible (low value cases: {low_count})")
        else:
            print(f"{name}: Stratification difficult (low value cases: {low_count})")
    
    # Select best fold
    print("\n=== Best Fold Selection ===")
    fold_metrics = calculate_fold_metrics(internal_df)
    best_fold, scores = select_best_checkpoint(fold_metrics)
    
    if best_fold is not None:
        print(f"Best Fold: {best_fold}")
        print("Scores per Fold:")
        for fold, score in scores.items():
            print(f"  Fold {fold}: MAE={fold_metrics[fold]['mae']:.4f}, STD={fold_metrics[fold]['std']:.4f}, Score={score:.4f}")
    else:
        print("Failed to calculate fold metrics")
    
    # Analyze Hb frequency
    print("\n=== Hb Frequency Analysis ===")
    freq_results = {}
    
    for name, hb in datasets.items():
        freq = analyze_hb_frequency(hb, name, hb_ranges)
        freq_results[name] = freq
        print(f"\n{name}:")
        for range_label, data in freq.items():
            print(f"  {range_label}: {data['count']} ({data['percentage']:.1f}%)")
    
    # Distribution comparison graph
    plt.figure(figsize=(15, 10))
    
    # Histogram
    plt.subplot(2, 2, 1)
    for name, hb in datasets.items():
        alpha = 0.7 if name in highlight_datasets else 0.4
        plt.hist(
            hb,
            bins=30,
            alpha=alpha,
            label=name,
            density=True,
            color=color_map.get(name, '#999999')
        )
    plt.xlabel('Hemoglobin (g/dL)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Hemoglobin Distribution Comparison')
    
    # Box plot
    plt.subplot(2, 2, 2)
    data_to_plot = [hb for hb in datasets.values()]
    box = plt.boxplot(data_to_plot, labels=list(datasets.keys()), patch_artist=True)
    for patch, name in zip(box['boxes'], datasets.keys()):
        patch.set_facecolor(color_map.get(name, '#999999'))
        patch.set_alpha(0.7 if name in highlight_datasets else 0.4)
    for median in box['medians']:
        median.set_color('#333333')
    plt.ylabel('Hemoglobin (g/dL)')
    plt.title('Hemoglobin Distribution Box Plot')
    
    # Bar plot for frequency by ranges
    plt.subplot(2, 2, 3)
    range_labels = [label for _, _, label in hb_ranges]
    x = np.arange(len(range_labels))
    width = 0.2
    
    for i, (name, freq) in enumerate(freq_results.items()):
        counts = [data['count'] for data in freq.values()]
        alpha = 0.85 if name in highlight_datasets else 0.5
        plt.bar(
            x + i*width,
            counts,
            width,
            label=name,
            alpha=alpha,
            color=color_map.get(name, '#999999')
        )
    
    plt.xlabel('Hb Range (g/dL)')
    plt.ylabel('Count')
    plt.xticks(x + width, range_labels, rotation=45)
    plt.legend()
    plt.title('Hb Frequency by Range')
    
    # Percentage plot
    plt.subplot(2, 2, 4)
    for i, (name, freq) in enumerate(freq_results.items()):
        percentages = [data['percentage'] for data in freq.values()]
        alpha = 0.85 if name in highlight_datasets else 0.5
        plt.bar(
            x + i*width,
            percentages,
            width,
            label=name,
            alpha=alpha,
            color=color_map.get(name, '#999999')
        )
    
    plt.xlabel('Hb Range (g/dL)')
    plt.ylabel('Percentage (%)')
    plt.xticks(x + width, range_labels, rotation=45)
    plt.legend()
    plt.title('Hb Percentage by Range')
    
    plt.tight_layout()
    plt.savefig('external_validation_proceed/hb_frequency_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical test (Internal vs External Master)
    print("\n=== Statistical Test (Internal vs External Master) ===")
    stat, p_value = stats.ks_2samp(internal_hb, external_master_hb)
    print(f"Kolmogorov-Smirnov test: statistic={stat:.4f}, p-value={p_value:.4f}")
    
    # Save results to Excel
    with pd.ExcelWriter('external_validation_proceed/ext_int_frequency_comparison_final.xlsx') as writer:
        # Frequency data
        freq_df = pd.DataFrame()
        for name, freq in freq_results.items():
            temp_df = pd.DataFrame.from_dict(freq, orient='index')
            temp_df = temp_df.reindex(range_labels)
            temp_df.columns = [f'{name}_count', f'{name}_percentage']
            if freq_df.empty:
                freq_df = temp_df
            else:
                freq_df = freq_df.join(temp_df, how='outer')

        freq_df = freq_df.reindex(range_labels)

        formatted_df = pd.DataFrame(index=freq_df.index)
        formatted_columns = []
        for name in freq_results.keys():
            count_col = f'{name}_count'
            pct_col = f'{name}_percentage'
            if count_col not in freq_df.columns:
                continue
            new_col = f'{name}_count (%)'
            counts = freq_df[count_col].fillna(0).astype(int)
            percentages = freq_df[pct_col].fillna(0.0)
            formatted_df[new_col] = [f"{count}({percentage:.2f})" for count, percentage in zip(counts, percentages)]
            formatted_columns.append(new_col)

        total_row = {}
        for name in freq_results.keys():
            count_col = f'{name}_count'
            formatted_col = f'{name}_count (%)'
            if count_col not in freq_df.columns:
                continue
            total_count = freq_df[count_col].sum()
            dataset_total = dataset_sizes.get(name, 0)
            total_percentage = (total_count / dataset_total * 100) if dataset_total else 0.0
            total_row[formatted_col] = f"{int(total_count)}({total_percentage:.2f})"

        formatted_df.loc['Total'] = total_row
        formatted_df = formatted_df.reset_index().rename(columns={'index': 'Hemoglobin range'})
        formatted_df = formatted_df[['Hemoglobin range'] + formatted_columns]

        formatted_df.to_excel(writer, sheet_name='Frequency', index=False)
        
        # Low value analysis
        low_hb_summary = pd.DataFrame({
            'Dataset': list(datasets.keys()),
            'Low_Hb_Count': [(hb < 8).sum() for hb in datasets.values()],
            'Total_Count': [len(hb) for hb in datasets.values()],
            'Percentage': [(hb < 8).sum() / len(hb) * 100 if len(hb) else 0 for hb in datasets.values()]
        })
        low_hb_summary.to_excel(writer, sheet_name='Low_Hb_Analysis', index=False)
        
        # Fold metrics
        if fold_metrics:
            fold_df = pd.DataFrame.from_dict(fold_metrics, orient='index')
            fold_df['score'] = [scores.get(fold, None) for fold in fold_metrics.keys()]
            fold_df.to_excel(writer, sheet_name='Fold_Metrics')
    
    print("\nAnalysis completed! Results saved in external_validation/ folder.")

if __name__ == "__main__":
    main()
