#!/usr/bin/env python3
"""
Image-bins Model Selection from Hyperparameter Optimization Results
================================================================

이 스크립트는 image-bins 하이퍼파라미터 최적화 결과를 분석하여 최적의 모델을 선택합니다.
Bayesian optimization으로 생성된 다양한 bins 값들 중 최고 성능을 가진 모델을 찾습니다.

작성일: 2025-09-22
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_analyze_image_bins_models():
    """combineResults_from_ImageBinsHPO 폴더의 모든 image-bins 모델을 분석"""
    print("🚀 Image-bins Model Selection from Bayesian Optimization Results")
    print("=" * 70)

    # 작업 디렉토리 설정
    os.chdir('/home/erdrajh/project_a6000/project25/2506Anemia/kimsangwon_code_3')

    # combineResults_from_ImageBinsHPO 폴더의 모든 파일 찾기
    results_pattern = 'combineResults_from_ImageBinsHPO/results_image-bins*_combined.xlsx'
    files = sorted(glob.glob(results_pattern))

    if not files:
        print(f"❌ 파일을 찾을 수 없습니다: {results_pattern}")
        return None

    print(f"📁 발견된 파일들: {len(files)}개")
    for i, file in enumerate(files, 1):
        bins_num = extract_bins_from_filename(file)
        print(f"   {i:2d}. image-bins{bins_num} ({os.path.basename(file)})")

    # 모든 파일 로드 및 분석
    all_results = []
    
    for file in files:
        print(f"\n📖 분석 중: {os.path.basename(file)}")
        result = analyze_single_file(file)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("❌ 분석할 수 있는 파일이 없습니다.")
        return None

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(all_results)
    
    return df

def extract_bins_from_filename(filename):
    """파일명에서 bins 값 추출"""
    import re
    match = re.search(r'image-bins(\d+)', filename)
    return int(match.group(1)) if match else 0

def analyze_single_file(file_path):
    """단일 파일 분석"""
    try:
        df = pd.read_excel(file_path)
        
        if 'ground truth' not in df.columns or 'prediction' not in df.columns:
            print(f"   ❌ 필요한 컬럼이 없습니다: {file_path}")
            return None

        # Bins 값 추출
        bins = extract_bins_from_filename(file_path)
        
        # 각 fold별 MAE 계산
        fold_maes = {}
        fold_counts = {}

        for fold in range(5):  # 0-4 folds
            fold_data = df[df['fold'] == fold]
            if len(fold_data) > 0:
                mae = np.mean(np.abs(fold_data['ground truth'] - fold_data['prediction']))
                fold_maes[fold] = mae
                fold_counts[fold] = len(fold_data)

        if not fold_maes:
            print(f"   ❌ fold 데이터가 없습니다: {file_path}")
            return None

        # 통계 계산
        fold_mae_values = list(fold_maes.values())
        mean_mae = np.mean(fold_mae_values)
        std_mae = np.std(fold_mae_values, ddof=1)
        
        # RMSE 계산
        rmse = np.sqrt(np.mean((df['ground truth'] - df['prediction'])**2))
        
        # CV (변동계수) 계산
        cv = std_mae / mean_mae if mean_mae > 0 else 0
        
        # 95% CI 계산
        if len(fold_mae_values) > 1:
            ci_range = stats.t.interval(0.95, len(fold_mae_values)-1, 
                                      loc=mean_mae, scale=stats.sem(fold_mae_values))
            ci_width = ci_range[1] - ci_range[0]
            ci_range_str = f"[{ci_range[0]:.4f}, {ci_range[1]:.4f}]"
        else:
            ci_width = 0
            ci_range_str = "[0.0000, 0.0000]"

        # Composite Score 계산 (성능 + 안정성 + 단순성)
        composite_score = mean_mae + 0.5 * std_mae + 0.1 * (bins / 100)

        result = {
            'Experiment': f'image-bins{bins}',
            'Bins': bins,
            'Mean_MAE': mean_mae,
            'Std_Dev': std_mae,
            'RMSE': rmse,
            'CV': cv,
            '95%_CI_Range': ci_range_str,
            'CI_Width': ci_width,
            'Composite_Score': composite_score,
            'Total_Samples': len(df),
            'Fold_Counts': fold_counts,
            'Fold_MAEs': fold_maes,
            'Source_File': file_path
        }

        print(f"   ✅ bins{bins}: MAE {mean_mae:.4f} ± {std_mae:.4f}")
        
        return result

    except Exception as e:
        print(f"   ❌ 분석 실패: {e}")
        return None

def analyze_best_models(df):
    """다양한 기준으로 최적 모델 분석"""
    print('\n📊 다양한 평가 지표로 Best Model 분석')
    print('=' * 60)

    # 1. MAE 기준 (Bayesian Optimization의 목표)
    print('\n1. 🎯 MAE 기준 (Bayesian Optimization 목표):')
    mae_best = df.loc[df['Mean_MAE'].idxmin()]
    print(f'   🏆 {mae_best["Experiment"]}')
    print(f'   📈 MAE: {mae_best["Mean_MAE"]:.4f} ± {mae_best["Std_Dev"]:.4f}')
    print(f'   🔢 Bins: {mae_best["Bins"]}')

    # 2. RMSE 기준
    print('\n2. 📏 RMSE 기준 (큰 오차에 더 큰 패널티):')
    rmse_best = df.loc[df['RMSE'].idxmin()]
    print(f'   🏆 {rmse_best["Experiment"]}')
    print(f'   📈 RMSE: {rmse_best["RMSE"]:.4f}')
    print(f'   🔢 Bins: {rmse_best["Bins"]}')

    # 3. Stability 기준
    print('\n3. 🎭 Stability 기준 (표준편차가 가장 작은 모델):')
    stability_best = df.loc[df['Std_Dev'].idxmin()]
    print(f'   🏆 {stability_best["Experiment"]}')
    print(f'   📊 Std: {stability_best["Std_Dev"]:.4f}')
    print(f'   📈 MAE: {stability_best["Mean_MAE"]:.4f}')
    print(f'   🔢 Bins: {stability_best["Bins"]}')

    # 4. CV 기준
    print('\n4. 📊 CV (변동계수) 기준:')
    cv_best = df.loc[df['CV'].idxmin()]
    print(f'   🏆 {cv_best["Experiment"]}')
    print(f'   📈 CV: {cv_best["CV"]:.4f}')
    print(f'   📈 MAE: {cv_best["Mean_MAE"]:.4f}')
    print(f'   🔢 Bins: {cv_best["Bins"]}')

    # 5. Composite Score
    print('\n5. 🏅 Composite Score (성능 + 안정성 + 단순성):')
    composite_best = df.loc[df['Composite_Score'].idxmin()]
    print(f'   🏆 {composite_best["Experiment"]}')
    print(f'   📊 Score: {composite_best["Composite_Score"]:.4f}')
    print(f'   📈 MAE: {composite_best["Mean_MAE"]:.4f}')
    print(f'   🔢 Bins: {composite_best["Bins"]}')

    # 6. CI Width 기준
    print('\n6. 🎯 95% CI 범위가 가장 좁은 모델 (가장 신뢰할 수 있는):')
    ci_best = df.loc[df['CI_Width'].idxmin()]
    print(f'   🏆 {ci_best["Experiment"]}')
    print(f'   📊 CI Width: {ci_best["CI_Width"]:.4f}')
    print(f'   📈 MAE: {ci_best["Mean_MAE"]:.4f}')
    print(f'   🔢 Bins: {ci_best["Bins"]}')

    # 7. 효율성 기준 (좋은 성능 + 최소 bins)
    print('\n7. ⚡ 효율성 기준 (좋은 성능 + 최소 bins):')
    # MAE가 상위 25% 안에 들면서 bins가 가장 작은 모델
    mae_threshold = df['Mean_MAE'].quantile(0.25)
    efficient_models = df[df['Mean_MAE'] <= mae_threshold]
    if not efficient_models.empty:
        efficient_best = efficient_models.loc[efficient_models['Bins'].idxmin()]
        print(f'   🏆 {efficient_best["Experiment"]}')
        print(f'   📈 MAE: {efficient_best["Mean_MAE"]:.4f} (상위 25%)')
        print(f'   🔢 Bins: {efficient_best["Bins"]} (최소)')
    else:
        efficient_best = df.loc[df['Bins'].idxmin()]
        print(f'   🏆 {efficient_best["Experiment"]} (최소 bins)')
        print(f'   📈 MAE: {efficient_best["Mean_MAE"]:.4f}')
        print(f'   🔢 Bins: {efficient_best["Bins"]}')

    return {
        'mae_best': mae_best,
        'rmse_best': rmse_best,
        'stability_best': stability_best,
        'cv_best': cv_best,
        'composite_best': composite_best,
        'ci_best': ci_best,
        'efficient_best': efficient_best
    }

def show_top_models_summary(df, top_n=5):
    """상위 모델들 요약 표시"""
    print(f'\n📋 Top {top_n} Models Summary')
    print('=' * 80)
    
    # MAE 기준으로 정렬
    top_models = df.nsmallest(top_n, 'Mean_MAE')
    
    print(f"{'Rank':<4} {'Model':<12} {'MAE':<8} {'Std':<8} {'RMSE':<8} {'CV':<8} {'Bins':<6} {'Composite':<10}")
    print('-' * 80)
    
    for i, (_, model) in enumerate(top_models.iterrows(), 1):
        print(f"{i:<4} {model['Experiment']:<12} {model['Mean_MAE']:<8.4f} "
              f"{model['Std_Dev']:<8.4f} {model['RMSE']:<8.4f} {model['CV']:<8.4f} "
              f"{model['Bins']:<6} {model['Composite_Score']:<10.4f}")

def analyze_bins_vs_performance_trend(df):
    """Bins 값과 성능의 관계 분석"""
    print(f'\n📈 Bins vs Performance 관계 분석')
    print('=' * 60)
    
    # Bins로 정렬
    df_sorted = df.sort_values('Bins')
    
    # 상관관계 분석
    bins_mae_corr = df['Bins'].corr(df['Mean_MAE'])
    bins_std_corr = df['Bins'].corr(df['Std_Dev'])
    
    print(f'📊 상관관계 분석:')
    print(f'   • Bins vs MAE 상관계수: {bins_mae_corr:.3f}')
    print(f'   • Bins vs Std 상관계수: {bins_std_corr:.3f}')
    
    print(f'\n📊 Bins 구간별 성능:')
    
    # Bins를 구간별로 나누어 분석
    bins_ranges = [
        (0, 20, "Very Low (0-20)"),
        (21, 40, "Low (21-40)"), 
        (41, 60, "Medium (41-60)"),
        (61, 80, "High (61-80)"),
        (81, 100, "Very High (81-100)")
    ]
    
    for min_bins, max_bins, label in bins_ranges:
        range_data = df[(df['Bins'] >= min_bins) & (df['Bins'] <= max_bins)]
        if not range_data.empty:
            avg_mae = range_data['Mean_MAE'].mean()
            avg_std = range_data['Std_Dev'].mean()
            best_in_range = range_data.loc[range_data['Mean_MAE'].idxmin()]
            print(f'   • {label}: Avg MAE {avg_mae:.4f}, Best: {best_in_range["Experiment"]} ({best_in_range["Mean_MAE"]:.4f})')

def final_recommendation(df, best_models):
    """최종 추천"""
    print(f'\n🎯 최종 추천 및 결론')
    print('=' * 60)
    
    # Bayesian Optimization의 목표인 MAE 기준 최고 모델
    mae_best = best_models['mae_best']
    
    print(f'🏆 Bayesian Optimization 목표 달성:')
    print(f'   ✅ 최고 성능 모델: {mae_best["Experiment"]}')
    print(f'   📈 MAE: {mae_best["Mean_MAE"]:.4f} ± {mae_best["Std_Dev"]:.4f}')
    print(f'   🔢 Bins: {mae_best["Bins"]}')
    print(f'   📊 전체 {len(df)}개 모델 중 최고 성능')
    
    # 성능 vs 효율성 trade-off 분석
    composite_best = best_models['composite_best']
    
    print(f'\n🎭 균형 고려 추천:')
    if mae_best['Experiment'] == composite_best['Experiment']:
        print(f'   ✅ 성능과 균형이 모두 최고인 모델: {mae_best["Experiment"]}')
        print(f'   💡 단일 모델로 최적 성능과 효율성을 동시 달성')
    else:
        print(f'   ⚖️ 균형 최고 모델: {composite_best["Experiment"]}')
        print(f'   📈 MAE: {composite_best["Mean_MAE"]:.4f} (vs 최고 {mae_best["Mean_MAE"]:.4f})')
        print(f'   🔢 Bins: {composite_best["Bins"]} (vs 최고 {mae_best["Bins"]})')
        
        mae_diff = composite_best["Mean_MAE"] - mae_best["Mean_MAE"]
        bins_diff = mae_best["Bins"] - composite_best["Bins"]
        
        print(f'   📊 Trade-off: MAE {mae_diff:+.4f} g/dL 손실로 {bins_diff} bins 절약')
    
    # 최종 추천
    print(f'\n💡 사용 시나리오별 추천:')
    print(f'   🎯 최고 성능 필요시: {mae_best["Experiment"]} (MAE {mae_best["Mean_MAE"]:.4f})')
    print(f'   ⚡ 효율성 고려시: {best_models["efficient_best"]["Experiment"]} (Bins {best_models["efficient_best"]["Bins"]})')
    print(f'   🎭 안정성 우선시: {best_models["stability_best"]["Experiment"]} (Std {best_models["stability_best"]["Std_Dev"]:.4f})')
    print(f'   🏅 종합 균형: {composite_best["Experiment"]} (Score {composite_best["Composite_Score"]:.4f})')

def improved_model_selection_for_image_bins(df):
    """
    개선된 기준(복잡도, 시나리오)을 적용하여 image-bins 모델을 평가하고 점수를 매깁니다.
    """
    # 1. 기본 지표 추가 계산
    if 'CV' not in df.columns:
        df['CV'] = df['Std_Dev'] / df['Mean_MAE']
    
    # CI_Width 계산 (95% CI 범위에서 추출)
    df['CI_Width'] = df['95%_CI_Range'].str.extract(r'\[([0-9.]+),\s*([0-9.]+)\]').astype(float).apply(
        lambda x: x[1] - x[0], axis=1
    )
    
    # 2. 🔥 Image-bins 전용 Complexity 계산
    # image-bins 모델은 단일 모달리티이므로 복잡도는 주로 bins 수에 의존
    df['Complexity'] = df['Bins'] / df['Bins'].max()
    
    # 3. 순위 계산 (낮을수록 좋음)
    rank_columns = ['Mean_MAE', 'Std_Dev', 'CV', 'CI_Width', 'Complexity']
    for col in rank_columns:
        df[f'{col}_Rank'] = df[col].rank(method='min')

    num_models = len(df)
    recommendations = {exp: {} for exp in df['Experiment']}

    # 4. 🔥 Image-bins 전용 시나리오별 가중치
    scenarios = {
        'performance_priority': {  # 연구/벤치마킹: 성능이 가장 중요
            'Mean_MAE': 0.8, 'Std_Dev': 0.1, 'Complexity': 0.1
        },
        'clinical_screening': {    # 🔥 임상 선별검사: 실용성(단순성)과 성능의 균형이 중요
            'Mean_MAE': 0.4, 'CV': 0.2, 'Complexity': 0.4
        },
        'stability_priority': {   # 안정성 우선: 예측의 일관성과 신뢰성이 가장 중요
            'Mean_MAE': 0.3, 'CV': 0.3, 'CI_Width': 0.4
        }
    }

    # 5. 점수 계산
    for scenario_name, weights in scenarios.items():
        for _, model in df.iterrows():
            score = 0
            for metric, weight in weights.items():
                rank_col = f'{metric}_Rank'
                # 순위를 0-1 사이의 점수로 변환 (1위가 가장 높은 점수)
                normalized_score = (num_models - model[rank_col] + 1) / num_models
                score += normalized_score * weight

            recommendations[model['Experiment']][scenario_name] = score

    return recommendations, df

def display_new_criteria_results(df, recommendations):
    """개선된 기준 분석 결과를 시나리오별로 정리하여 출력합니다."""

    scenarios_desc = {
        'performance_priority': '성능 우선 (연구/벤치마킹)',
        'clinical_screening': '임상 선별검사 (실용성 중시)',
        'stability_priority': '안정성 우선 (신뢰성 중시)'
    }

    print("\n\n" + "=" * 80)
    print("🎯 시나리오별 최적 Image-bins 모델 추천 (개선된 기준 적용)")
    print("=" * 80)

    # 각 시나리오별로 상위 3개 모델을 출력합니다.
    for scenario_key, scenario_desc in scenarios_desc.items():
        # 시나리오별 점수를 기준으로 모델 정렬
        scores = {exp: rec[scenario_key] for exp, rec in recommendations.items()}
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n🏆 시나리오: {scenario_desc}")
        print("-" * 60)

        for i, (model_name, score) in enumerate(sorted_models[:3]):
            model_row = df[df['Experiment'] == model_name].iloc[0]
            marker = "⭐" if i == 0 else f"  {i+1}."
            print(f"{marker} {model_name}")
            print(f"     - 종합 점수: {score:.4f}")
            print(f"     - MAE: {model_row['Mean_MAE']:.4f} (Rank: {int(model_row['Mean_MAE_Rank'])})")
            print(f"     - Std Dev: {model_row['Std_Dev']:.4f} (Rank: {int(model_row['Std_Dev_Rank'])})")
            print(f"     - Complexity: {model_row['Complexity']:.3f} (Rank: {int(model_row['Complexity_Rank'])})")

    # 🔥 Bayesian Optimization 목표에 따른 최종 추천 모델 (MAE 최소)
    mae_best_model = df.loc[df['Mean_MAE'].idxmin(), 'Experiment']

    print("\n\n" + "=" * 80)
    print("✅ 최종 Bayesian Optimization 추천 Image-bins 모델")
    print("=" * 80)
    print(f"🎉 {mae_best_model}")
    print("   - 이 모델은 Bayesian optimization의 목표인 'MAE 최소화'를 달성한 최고 성능 모델입니다.")
    print("   - 44개 image-bins 모델 중 가장 낮은 MAE를 기록했습니다.")

    return mae_best_model

def create_hpo_table_with_new_scores(df, recommendations):
    """HPO 스타일의 테이블을 새로운 점수와 함께 생성"""
    # recommendations 딕셔너리를 데이터프레임으로 변환
    scores_df = pd.DataFrame.from_dict(recommendations, orient='index')
    scores_df.rename(columns={
        'performance_priority': 'Score (Performance)',
        'clinical_screening': 'Score (Clinical)', 
        'stability_priority': 'Score (Stability)'
    }, inplace=True)

    # 원본 데이터프레임과 점수 데이터프레임 병합
    final_df = df.merge(scores_df, left_on='Experiment', right_index=True, how='left')
    
    # HPO 테이블 스타일로 변환 (fold별 MAE 추가)
    hpo_style_results = []
    
    for _, row in final_df.iterrows():
        fold_maes = row['Fold_MAEs']
        
        result_row = {
            'Experiment': row['Experiment'],
            '1st Fold': round(fold_maes[0], 4) if len(fold_maes) > 0 else None,
            '2nd Fold': round(fold_maes[1], 4) if len(fold_maes) > 1 else None,
            '3rd Fold': round(fold_maes[2], 4) if len(fold_maes) > 2 else None,
            '4th Fold': round(fold_maes[3], 4) if len(fold_maes) > 3 else None,
            '5th Fold': round(fold_maes[4], 4) if len(fold_maes) > 4 else None,
            'Mean_MAE': round(row['Mean_MAE'], 4),
            'Std_Dev': round(row['Std_Dev'], 4),
            '95%_CI_Range': row['95%_CI_Range'],
            'Model': 'image-bins',
            'Bins': row['Bins'],
            'CV': round(row['CV'], 4),
            'CI_Width': round(row['CI_Width'], 4),
            'Complexity': round(row['Complexity'], 4),
            'Score (Performance)': round(row['Score (Performance)'], 4),
            'Score (Clinical)': round(row['Score (Clinical)'], 4),
            'Score (Stability)': round(row['Score (Stability)'], 4),
            'Mean_MAE_Rank': int(row['Mean_MAE_Rank']),
            'Std_Dev_Rank': int(row['Std_Dev_Rank']),
            'Complexity_Rank': int(row['Complexity_Rank'])
        }
        hpo_style_results.append(result_row)
    
    hpo_df = pd.DataFrame(hpo_style_results)
    # MAE 기준으로 정렬
    hpo_df = hpo_df.sort_values('Mean_MAE')
    
    return hpo_df

def create_scatter_plot_grayscale(df, output_dir):
    """Image-bins HPO 결과의 grayscale scatter plot 생성"""
    plt.figure(figsize=(15, 10))
    
    # Bins 값과 MAE로 scatter plot 생성
    bins_values = df['Bins']
    mae_values = df['Mean_MAE']
    std_values = df['Std_Dev']
    
    # CI 값 추출
    ci_lower = df['95%_CI_Range'].str.extract(r'\[([0-9.]+),').astype(float).values.flatten()
    ci_upper = df['95%_CI_Range'].str.extract(r', ([0-9.]+)\]').astype(float).values.flatten()
    
    # Grayscale scatter plot with error bars
    plt.errorbar(bins_values, mae_values, yerr=std_values,
                fmt='o', capsize=5, label='Image-bins (Std Dev)',
                color='black', alpha=0.7)
    
    # Add CI as filled area
    plt.fill_between(bins_values, ci_lower, ci_upper,
                   alpha=0.2, color='gray',
                   label='Image-bins (95% CI)')
    
    plt.xlabel('Bins', fontsize=12)
    plt.ylabel('Mean MAE', fontsize=12)
    plt.title('Image-bins Hyperparameter Optimization: Bins vs Mean MAE', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save grayscale plot
    output_path = output_dir / 'hyperparameter_optimization_scatter_grayscale_imageBinsOnly.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'   📊 Grayscale plot: {output_path}')

def save_enhanced_results(df, best_models, recommendations, best_overall_model):
    """향상된 결과를 종합하여 저장"""
    output_dir = Path("report_250924")
    output_dir.mkdir(exist_ok=True)
    
    # 1. HPO 스타일 테이블 with new scores 생성 및 저장
    hpo_table = create_hpo_table_with_new_scores(df, recommendations)
    excel_path = output_dir / 'spp_table1_full_hyperparameter_search_mae_imageBinsOnly.xlsx'
    hpo_table.to_excel(excel_path, index=False)
    
    # 2. Grayscale scatter plot 생성
    create_scatter_plot_grayscale(df, output_dir)
    
    # 3. 기존 요약 결과들도 저장
    df_output = df.copy()
    df_output = df_output.sort_values('Mean_MAE')
    df_output.to_excel(output_dir / 'image_bins_optimization_results_summary.xlsx', index=False)
    
    # Best models 요약 저장
    best_summary = []
    for key, model in best_models.items():
        best_summary.append({
            'Criteria': key.replace('_', ' ').title(),
            'Model': model['Experiment'],
            'MAE': model['Mean_MAE'],
            'Std_Dev': model['Std_Dev'],
            'Bins': model['Bins'],
            'Composite_Score': model['Composite_Score']
        })
    
    best_df = pd.DataFrame(best_summary)
    best_df.to_excel(output_dir / 'image_bins_best_models_summary.xlsx', index=False)
    
    print(f'\n💾 향상된 결과 저장 완료:')
    print(f'   📊 HPO 스타일 테이블: {excel_path}')
    print(f'   📊 전체 결과: {output_dir}/image_bins_optimization_results_summary.xlsx')
    print(f'   🏆 Best models: {output_dir}/image_bins_best_models_summary.xlsx')
    print(f'   🎯 최종 추천 모델: {best_overall_model}')

def main():
    """메인 실행 함수"""
    
    # 1. 모든 image-bins 모델 로드 및 분석
    df = load_and_analyze_image_bins_models()
    if df is None:
        return
    
    # 2. 다양한 기준으로 최적 모델 분석
    best_models = analyze_best_models(df)
    
    # 3. 상위 모델들 요약
    show_top_models_summary(df)
    
    # 4. Bins vs Performance 관계 분석
    analyze_bins_vs_performance_trend(df)
    
    # 5. 최종 추천
    final_recommendation(df, best_models)
    
    # 6. 🔥 개선된 기준으로 모델 평가
    recommendations, enhanced_df = improved_model_selection_for_image_bins(df)
    
    # 7. 🔥 시나리오별 분석 결과 출력
    best_overall_model = display_new_criteria_results(enhanced_df, recommendations)
    
    # 8. 🔥 향상된 결과 저장 (HPO 테이블 + Grayscale plot)
    save_enhanced_results(enhanced_df, best_models, recommendations, best_overall_model)
    
    print(f"\n🎉 Image-bins 모델 선택 분석 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()