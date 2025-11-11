import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Seaborn 스타일 설정
sns.set_style('whitegrid')

def adaptive_few_shot_sampling(df, country_config):
    """적응적 구간별 샘플링"""
    
    # 우선순위 구간 정의 (임상적 중요도순)
    priority_bins = [
        (3, 7),   # 매우 중요: 수혈 기준
        (7, 10),  # 중요: 빈혈 범위
        (13, 16), # 중요: 고수치
        (10, 13)  # 일반: 정상 범위
    ]
    
    target = 10 if country_config != 'joint' else 20
    selected = []
    
    # 1단계: 우선순위 구간에서 available samples 확보
    for bin_range in priority_bins:
        bin_data = df[(df['Hb'] >= bin_range[0]) & (df['Hb'] < bin_range[1])]
        
        if len(bin_data) > 0:
            # 구간 크기에 비례하여 샘플 수 결정
            max_from_bin = min(3, len(bin_data))  # 구간당 최대 3개
            n_samples = min(max_from_bin, target - len(selected))
            
            if n_samples > 0:
                sampled = bin_data.sample(n=n_samples, random_state=42)
                selected.extend(sampled.index.tolist())
        
        if len(selected) >= target:
            break
    
    # 2단계: 부족분은 weighted random sampling
    if len(selected) < target:
        remaining = target - len(selected)
        available = df.index.difference(selected)
        
        if len(available) >= remaining:
            # Hb 분포를 고려한 가중 샘플링
            weights = calculate_clinical_weights(df.loc[available]['Hb'])
            additional = df.loc[available].sample(
                n=remaining, 
                weights=weights,
                random_state=42
            )
            selected.extend(additional.index.tolist())
    
    return selected[:target]

def calculate_clinical_weights(hb_values):
    """임상적 중요도 기반 가중치"""
    weights = []
    for hb in hb_values:
        if hb < 7:
            weight = 3.0    # 매우 중요
        elif hb < 10:
            weight = 2.0    # 중요
        elif hb > 15:
            weight = 2.0    # 중요
        else:
            weight = 1.0    # 기본
        weights.append(weight)
    
    return weights

# 전략 정의
india_strategy = {
    'target_distribution': {
        (7, 10): 3,   # 빈혈 범위 확보
        (10, 13): 4,  # 기존 강점 활용
        (13, 16): 3   # 고수치 보완
    }
}

italy_strategy = {
    'target_distribution': {
        (7, 10): 4,   # 저수치 최대한 확보
        (10, 13): 3,  # 중간값 보완
        (13, 16): 3   # 기존 데이터 활용
    }
}

joint_strategy = {
    'per_country': {
        'India': italy_strategy,  # 상호 보완
        'Italy': india_strategy
    }
}

def calculate_fold_metrics(internal_df):
    """Fold별 MAE, STD 계산"""
    fold_metrics = {}
    for fold in range(5):  # 0-4 fold
        fold_data = internal_df[internal_df['fold'] == fold]
        if len(fold_data) > 0:
            mae = np.mean(np.abs(fold_data['ground truth'] - fold_data['prediction']))
            std = np.std(np.abs(fold_data['ground truth'] - fold_data['prediction']))
            fold_metrics[fold] = {'mae': mae, 'std': std}
    return fold_metrics

def select_best_checkpoint(fold_metrics):
    """Composite score를 사용하여 best fold 선택"""
    if not fold_metrics:
        return None, None
    
    # 모든 fold의 값들 수집
    maes = [fold_metrics[fold]['mae'] for fold in fold_metrics]
    stds = [fold_metrics[fold]['std'] for fold in fold_metrics]
    cvs = [fold_metrics[fold]['std'] / fold_metrics[fold]['mae'] for fold in fold_metrics]
    
    # 정규화를 위한 min/max 계산
    mae_min, mae_max = min(maes), max(maes)
    std_max = max(stds)
    cv_max = max(cvs)
    
    scores = {}
    for fold in fold_metrics:
        mae = fold_metrics[fold]['mae']
        std = fold_metrics[fold]['std']
        cv = std / mae
        
        # 정규화
        mae_norm = (mae - mae_min) / (mae_max - mae_min) if mae_max != mae_min else 0
        std_norm = std / std_max if std_max != 0 else 0
        cv_norm = cv / cv_max if cv_max != 0 else 0
        
        # External validation에 최적화된 가중치
        score = 0.5 * mae_norm + 0.3 * std_norm + 0.2 * cv_norm
        scores[fold] = score
    
    best_fold = min(scores, key=scores.get)
    return best_fold, scores

def analyze_hb_frequency(hb_values, label, hb_ranges):
    """Hb 빈도 분석 - 명확한 구간 정의"""
    freq = {}
    total = len(hb_values)
    
    for i, (lower, upper, label_range) in enumerate(hb_ranges):
        if lower is None:  # <6
            count = (hb_values < upper).sum()
        elif upper is None:  # >=15
            count = (hb_values >= lower).sum()
        else:  # 명확한 범위: lower <= Hb < upper
            count = ((hb_values >= lower) & (hb_values < upper)).sum()
        
        freq[label_range] = {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }
    
    return freq

def main():
    # Hb 범위 정의 - 명확한 구간: (lower, upper, label)
    # lower <= Hb < upper (양쪽 경계 명확)
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
    
    # Internal 데이터 로드
    internal_df = pd.read_excel('results_excel_file_final/results_image-bins76_combined.xlsx')
    internal_hb = internal_df['ground truth']
    
    # External 데이터 로드
    external_master_df = pd.read_csv('external_validation_joint_results/external_validation_master.csv')
    external_india_df = pd.read_csv('external_validation_joint_results/external_validation_india.csv')
    external_italy_df = pd.read_csv('external_validation_joint_results/external_validation_italy.csv')
    # external_ghana_df = pd.read_excel('external_validation_ghana/anemiaDataGhana.xlsx')
    # print(f"Ghana data columns: {external_ghana_df.columns.tolist()}")
    # print(f"Ghana data sample:\n{external_ghana_df.head()}")
    
    # Hb 열 처리: 빈 문자열과 "_" 제외
    external_master_df = external_master_df[external_master_df['Hb'] != ""]
    external_master_df = external_master_df[external_master_df['Hb'] != "_"]
    external_india_df = external_india_df[external_india_df['Hb'] != ""]
    external_india_df = external_india_df[external_india_df['Hb'] != "_"]
    external_italy_df = external_italy_df[external_italy_df['Hb'] != ""]
    external_italy_df = external_italy_df[external_italy_df['Hb'] != "_"]
    # Ghana 데이터 처리 (사용하지 않음)
    # if 'Hb' in external_ghana_df.columns:
    #     external_ghana_df = external_ghana_df[external_ghana_df['Hb'] != ""]
    #     external_ghana_df = external_ghana_df[external_ghana_df['Hb'] != "_"]
    
    external_master_hb = pd.to_numeric(external_master_df['Hb'], errors='coerce').dropna()
    external_india_hb = pd.to_numeric(external_india_df['Hb'], errors='coerce').dropna()
    external_italy_hb = pd.to_numeric(external_italy_df['Hb'], errors='coerce').dropna()
    # external_ghana_hb = pd.to_numeric(external_ghana_df['Hb'], errors='coerce').dropna() if 'Hb' in external_ghana_df.columns else pd.Series(dtype=float)
    
    # 저수치 Hb 케이스 분석 (<8 g/dL)
    print("=== 저수치 Hb 케이스 분석 (<8 g/dL) ===")
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
        'Internal': '#1f77b4',          # blue (강조)
        'External (Master)': '#ff7f0e', # orange (강조)
        'External (India)': '#aaaaaa',  # medium-light gray (부드러움)
        'External (Italy)': '#666666',  # dark gray (진하고 잘 보임)
    # 'External (Ghana)': '#2ca02c'   # green
    }
    
    for name, hb in datasets.items():
        low_count = (hb < 8).sum()
        total = len(hb)
        print(f"{name}: {low_count}/{total} ({low_count/total*100:.1f}%)")
    
    # 층화 가능성 평가
    print("\n=== 층화 가능성 평가 ===")
    for name, hb in datasets.items():
        low_count = (hb < 8).sum()
        if low_count >= 5:  # 최소 5개 이상이면 층화 가능
            print(f"{name}: 층화 가능 (저수치 케이스 {low_count}개)")
        else:
            print(f"{name}: 층화 어려움 (저수치 케이스 {low_count}개)")
    
    # Best fold 선택
    print("\n=== Best Fold 선택 ===")
    fold_metrics = calculate_fold_metrics(internal_df)
    best_fold, scores = select_best_checkpoint(fold_metrics)
    
    if best_fold is not None:
        print(f"Best Fold: {best_fold}")
        print("Fold별 Scores:")
        for fold, score in scores.items():
            print(f"  Fold {fold}: MAE={fold_metrics[fold]['mae']:.4f}, STD={fold_metrics[fold]['std']:.4f}, Score={score:.4f}")
    else:
        print("Fold metrics 계산 실패")
    
    # Hb 빈도 분석
    print("\n=== Hb 빈도 분석 ===")
    freq_results = {}
    
    for name, hb in datasets.items():
        freq = analyze_hb_frequency(hb, name, hb_ranges)
        freq_results[name] = freq
        print(f"\n{name}:")
        for range_label, data in freq.items():
            print(f"  {range_label}: {data['count']} ({data['percentage']:.1f}%)")
    
    # 분포 비교 그래프
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
    
    # 통계적 검정 (Internal vs External Master)
    print("\n=== 통계적 검정 (Internal vs External Master) ===")
    stat, p_value = stats.ks_2samp(internal_hb, external_master_hb)
    print(f"Kolmogorov-Smirnov test: statistic={stat:.4f}, p-value={p_value:.4f}")
    
    # 결과 Excel 저장
    with pd.ExcelWriter('external_validation_proceed/ext_int_frequency_comparison_final.xlsx') as writer:
        # 빈도 데이터
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
        
        # 저수치 분석
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
    
    print("\n분석 완료! 결과는 external_validation/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
