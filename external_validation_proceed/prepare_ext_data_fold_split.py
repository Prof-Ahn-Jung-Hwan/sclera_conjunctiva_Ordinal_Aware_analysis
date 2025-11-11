import pandas as pd
import os
import numpy as np

# --------------------------------------------------------------------------
# 1. 샘플링 로직 함수 (수정된 전략)
# --------------------------------------------------------------------------

def eyedye_india_balanced_sampling(df_india, target_size, random_seed):
    """India 데이터셋을 위한 균등 샘플링 (전 구간 균등)"""
    selected_indices = []
    
    # 우선순위 구간 정의
    priority_bins = [
        (7, 10),   # 7-10
        (10, 13),  # 10-13
        (13, 16)   # 13-16
    ]
    
    samples_per_bin = target_size // len(priority_bins)
    remaining_samples = target_size % len(priority_bins)
    
    np.random.seed(random_seed)
    
    for i, (low, high) in enumerate(priority_bins):
        bin_data = df_india[(df_india['Hb'] >= low) & (df_india['Hb'] < high)]
        
        # 각 구간에 균등하게 할당 + 나머지 첫 번째 구간에 추가
        target_for_bin = samples_per_bin + (1 if i == 0 and remaining_samples > 0 else 0)
        
        if len(bin_data) > 0:
            n_sample = min(target_for_bin, len(bin_data))
            sampled = bin_data.sample(n=n_sample, random_state=random_seed + i)
            selected_indices.extend(sampled.index.tolist())
    
    # 목표 수량에 미달 시, 전체에서 추가 샘플링
    remaining_needed = target_size - len(selected_indices)
    if remaining_needed > 0:
        available = df_india.index.difference(selected_indices)
        if len(available) > 0:
            additional = df_india.loc[available].sample(
                n=min(remaining_needed, len(available)), 
                random_state=random_seed + 100
            )
            selected_indices.extend(additional.index.tolist())
    
    return selected_indices[:target_size]

def eyedye_italy_compensatory_sampling(df_italy, target_size, random_seed):
    """Italy 데이터셋을 위한 보상적 샘플링 (저수치 최우선)"""
    selected_indices = []
    
    # 우선순위 구간 정의 (저수치부터 최우선)
    priority_bins = [
        (0, 9),    # <9 (최우선)
        (9, 13),   # 9-13
        (13, 15),  # 13-15 (극고수치 최소화)
    ]
    
    # 우선순위에 따른 샘플 수 할당 (저수치에 더 많은 비중)
    allocation_ratio = [0.5, 0.3, 0.2]  # <9에 50%, 9-13에 30%, 13-15에 20%
    
    np.random.seed(random_seed)
    
    for i, ((low, high), ratio) in enumerate(zip(priority_bins, allocation_ratio)):
        if len(selected_indices) >= target_size:
            break
            
        bin_data = df_italy[(df_italy['Hb'] >= low) & (df_italy['Hb'] < high)]
        target_for_bin = int(target_size * ratio)
        
        if len(bin_data) > 0:
            n_sample = min(target_for_bin, len(bin_data), target_size - len(selected_indices))
            sampled = bin_data.sample(n=n_sample, random_state=random_seed + i)
            selected_indices.extend(sampled.index.tolist())
    
    # 목표 수량에 미달 시, 저수치 우선으로 추가 샘플링
    remaining_needed = target_size - len(selected_indices)
    if remaining_needed > 0:
        available = df_italy.index.difference(selected_indices)
        if len(available) > 0:
            # 저수치 우선으로 가중치 부여
            available_data = df_italy.loc[available]
            weights = np.where(available_data['Hb'] < 9, 3.0,  # <9에 높은 가중치
                      np.where(available_data['Hb'] < 13, 2.0,  # 9-13에 중간 가중치
                               1.0))  # 나머지에 낮은 가중치
            
            additional = available_data.sample(
                n=min(remaining_needed, len(available)), 
                weights=weights,
                random_state=random_seed + 200
            )
            selected_indices.extend(additional.index.tolist())
    
    return selected_indices[:target_size]

def ghana_sampling(df_ghana, target_size, random_seed):
    """Ghana 데이터셋을 위한 샘플링"""
    # 기존 Ghana 샘플링 로직 유지
    return sample_ghana_stratified(df_ghana, target_size, random_seed)

def sample_ghana_stratified(df, target, random_seed):
    """Ghana 데이터셋을 위한 우선순위 기반 층화 샘플링"""
    selected_indices = []
    
    # 우선순위 및 목표 샘플 수 정의
    priority_bins = {
        (0, 6): 8,
        (6, 8): 12,
        (8, 10): 15,
        (10, 12): 25,
        (12, 14): 25,
        (14, 16): 15
    }
    
    df_remaining = df.copy()
    np.random.seed(random_seed)

    for (low, high), n_target in priority_bins.items():
        if len(selected_indices) >= target:
            break
        
        bin_data = df_remaining[(df_remaining['Hb'] >= low) & (df_remaining['Hb'] < high)]
        n_sample = min(n_target, len(bin_data), target - len(selected_indices))
        
        if n_sample > 0:
            sampled = bin_data.sample(n=n_sample, random_state=random_seed)
            selected_indices.extend(sampled.index.tolist())
            df_remaining = df_remaining.drop(sampled.index)

    # 목표 수량에 미달 시, 가중 샘플링으로 채움
    remaining_needed = target - len(selected_indices)
    if remaining_needed > 0 and not df_remaining.empty:
        weights = calculate_clinical_weights(df_remaining['Hb'])
        additional = df_remaining.sample(
            n=min(remaining_needed, len(df_remaining)),
            weights=weights,
            random_state=random_seed + 300
        )
        selected_indices.extend(additional.index.tolist())
        
    return selected_indices[:target]

def calculate_clinical_weights(hb_values):
    """임상적 중요도 기반 가중치 (저수치/고수치 강조)"""
    weights = np.ones(len(hb_values))
    hb_series = pd.Series(hb_values)
    weights[hb_series < 7] = 4.0
    weights[hb_series < 10] = 2.0
    weights[hb_series > 15] = 2.0
    return weights

# --------------------------------------------------------------------------
# 2. 메인 실행 로직
# --------------------------------------------------------------------------

def main():
    print("--- External Validation 데이터 생성을 시작합니다. ---")

    # --- 설정값 ---
    RANDOM_SEED = 42
    
    # Eyedye 할당 전략
    eyedye_allocation = {
        'India': {
            'target': 18,
            'strategy': 'balanced_sampling',  # 전 구간 균등
            'priority': ['7-10', '10-13', '13-16']
        },
        'Italy': {
            'target': 12, 
            'strategy': 'compensatory_sampling',  # 부족한 저수치 최우선
            'priority': ['<9', '9-13', '13-15']  # 극고수치 최소화
        }
    }
    
    # Ghana 할당
    ghana_target = 100  # 708의 14.1%
    
    # 입력 파일 경로
    INPUT_CSVS = {
        'eyedye': 'external_validation_joint_results/external_validation_master.csv',
        'ghana': 'external_validation_ghana/anemiaDataGhana.xlsx'
    }
    
    # 출력 폴더 생성
    output_folder = 'external_validation_proceed'
    os.makedirs(output_folder, exist_ok=True)
    
    # --- Eyedye 데이터 처리 ---
    print(f"\n[EYEDYE] 데이터셋 처리 중...")
    df_eyedye = pd.read_csv(INPUT_CSVS['eyedye'])
    
    # Hb 값 전처리
    print("Hb 값 전처리 중...")
    # 빈 문자열을 NaN으로 변환
    df_eyedye['Hb'] = df_eyedye['Hb'].replace("", np.nan)
    # "_" 포함된 행 제거
    df_eyedye = df_eyedye[df_eyedye['Hb'] != "_"]
    # float로 변환
    df_eyedye['Hb'] = pd.to_numeric(df_eyedye['Hb'], errors='coerce')
    # NaN 제거
    df_eyedye = df_eyedye.dropna(subset=['Hb'])
    
    # full_path 경로 수정 (exteranal_validation -> external_validation_joint_results)
    # 원본 데이터에 오타가 있어서 exteranal_validation으로 되어있음
    df_eyedye['full_path'] = df_eyedye['full_path'].str.replace(
        'exteranal_validation/', 
        'external_validation_joint_results/', 
        regex=False
    )
    # 혹시 정상 경로도 있을 수 있으니 추가로 처리
    df_eyedye['full_path'] = df_eyedye['full_path'].str.replace(
        'external_validation/', 
        'external_validation_joint_results/', 
        regex=False
    )
    
    print(f"전처리 후 Eyedye 데이터: {len(df_eyedye)}개 샘플")
    print(f"India: {len(df_eyedye[df_eyedye['Country'] == 'India'])}개")
    print(f"Italy: {len(df_eyedye[df_eyedye['Country'] == 'Italy'])}개")
    
    # 국가별 분리
    df_india = df_eyedye[df_eyedye['Country'] == 'India'].copy()
    df_italy = df_eyedye[df_eyedye['Country'] == 'Italy'].copy()
    
    # 국가별 샘플링
    india_train_indices = eyedye_india_balanced_sampling(df_india, eyedye_allocation['India']['target'], RANDOM_SEED)
    italy_train_indices = eyedye_italy_compensatory_sampling(df_italy, eyedye_allocation['Italy']['target'], RANDOM_SEED)
    
    # 전체 train indices 합치기
    eyedye_train_indices = india_train_indices + italy_train_indices
    
    # Train/Test 분할
    df_eyedye_train = df_eyedye.loc[eyedye_train_indices]
    df_eyedye_test = df_eyedye.drop(eyedye_train_indices)
    
    # 저장
    eyedye_train_path = os.path.join(output_folder, 'train_ext_eyedye.csv')
    eyedye_test_path = os.path.join(output_folder, 'test_ext_eyedye.csv')
    
    df_eyedye_train.to_csv(eyedye_train_path, index=False)
    df_eyedye_test.to_csv(eyedye_test_path, index=False)
    
    print(f"Eyedye Train Set: {len(df_eyedye_train)}개 샘플 ({len(df_eyedye_train)/len(df_eyedye)*100:.1f}%)")
    print(f"  - India: {len(df_eyedye_train[df_eyedye_train['Country'] == 'India'])}개")
    print(f"  - Italy: {len(df_eyedye_train[df_eyedye_train['Country'] == 'Italy'])}개")
    print(f"Eyedye Test Set: {len(df_eyedye_test)}개 샘플")
    print(f"저장 완료: {eyedye_train_path}, {eyedye_test_path}")
    
    # --- Ghana 데이터 처리 ---
    print(f"\n[GHANA] 데이터셋 처리 중...")
    df_ghana = pd.read_excel(INPUT_CSVS['ghana'])
    
    # Hb 값 전처리
    print("Hb 값 전처리 중...")
    # 빈 문자열을 NaN으로 변환
    df_ghana['Hb'] = df_ghana['Hb'].replace("", np.nan)
    # "_" 포함된 행 제거
    df_ghana = df_ghana[df_ghana['Hb'] != "_"]
    # float로 변환
    df_ghana['Hb'] = pd.to_numeric(df_ghana['Hb'], errors='coerce')
    # NaN 제거
    df_ghana = df_ghana.dropna(subset=['Hb'])
    
    # Ghana 데이터에 full_path 추가
    img_base_path = 'external_validation_ghana/total_ghana_img'
    df_ghana['full_path'] = df_ghana['w_filename'].apply(lambda x: os.path.join(img_base_path, f"{x}.png"))
    
    print(f"전처리 후 Ghana 데이터: {len(df_ghana)}개 샘플")
    
    # Ghana 샘플링
    ghana_train_indices = ghana_sampling(df_ghana, ghana_target, RANDOM_SEED)
    
    # Train/Test 분할
    df_ghana_train = df_ghana.loc[ghana_train_indices]
    df_ghana_test = df_ghana.drop(ghana_train_indices)
    
    # 저장
    ghana_train_path = os.path.join(output_folder, 'train_ext_ghana.csv')
    ghana_test_path = os.path.join(output_folder, 'test_ext_ghana.csv')
    
    df_ghana_train.to_csv(ghana_train_path, index=False)
    df_ghana_test.to_csv(ghana_test_path, index=False)
    
    print(f"Ghana Train Set: {len(df_ghana_train)}개 샘플 ({len(df_ghana_train)/len(df_ghana)*100:.1f}%)")
    print(f"Ghana Test Set: {len(df_ghana_test)}개 샘플")
    print(f"저장 완료: {ghana_train_path}, {ghana_test_path}")
    
    # --- 요약 정보 출력 ---
    print(f"\n--- 생성 완료 요약 ---")
    print(f"Eyedye: {len(df_eyedye_train)}/{len(df_eyedye)} = {len(df_eyedye_train)/len(df_eyedye)*100:.1f}%")
    print(f"Ghana: {len(df_ghana_train)}/{len(df_ghana)} = {len(df_ghana_train)/len(df_ghana)*100:.1f}%")
    print(f"모든 파일이 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()