import pandas as pd
import os
import numpy as np

# --------------------------------------------------------------------------
# 1. Sampling logic functions (modified strategy)
# --------------------------------------------------------------------------

def eyedye_india_balanced_sampling(df_india, target_size, random_seed):
    """Uniform sampling for India dataset (uniform across all ranges)"""
    selected_indices = []
    
    # Define priority ranges
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
        
        # Allocate equally to each range + add remainder to first range
        target_for_bin = samples_per_bin + (1 if i == 0 and remaining_samples > 0 else 0)
        
        if len(bin_data) > 0:
            n_sample = min(target_for_bin, len(bin_data))
            sampled = bin_data.sample(n=n_sample, random_state=random_seed + i)
            selected_indices.extend(sampled.index.tolist())
    
    # If below target quantity, additional sampling from entire dataset
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
    """Compensatory sampling for Italy dataset (low values prioritized)"""
    selected_indices = []
    
    # Define priority ranges (low values prioritized first)
    priority_bins = [
        (0, 9),    # <9 (highest priority)
        (9, 13),   # 9-13
        (13, 15),  # 13-15 (minimize extreme high values)
    ]
    
    # Allocate sample numbers according to priority (more weight on low values)
    allocation_ratio = [0.5, 0.3, 0.2]  # 50% for <9, 30% for 9-13, 20% for 13-15
    
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
    
    # If below target quantity, additional sampling prioritizing low values
    remaining_needed = target_size - len(selected_indices)
    if remaining_needed > 0:
        available = df_italy.index.difference(selected_indices)
        if len(available) > 0:
            # Assign weights prioritizing low values
            available_data = df_italy.loc[available]
            weights = np.where(available_data['Hb'] < 9, 3.0,  # High weight for <9
                      np.where(available_data['Hb'] < 13, 2.0,  # Medium weight for 9-13
                               1.0))  # Low weight for others
            
            additional = available_data.sample(
                n=min(remaining_needed, len(available)), 
                weights=weights,
                random_state=random_seed + 200
            )
            selected_indices.extend(additional.index.tolist())
    
    return selected_indices[:target_size]

def ghana_sampling(df_ghana, target_size, random_seed):
    """Sampling for Ghana dataset"""
    # Keep existing Ghana sampling logic
    return sample_ghana_stratified(df_ghana, target_size, random_seed)

def sample_ghana_stratified(df, target, random_seed):
    """Priority-based stratified sampling for Ghana dataset"""
    selected_indices = []
    
    # Define priority and target sample numbers
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

    # If below target quantity, fill with weighted sampling
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
    """Clinical importance-based weights (emphasize low/high values)"""
    weights = np.ones(len(hb_values))
    hb_series = pd.Series(hb_values)
    weights[hb_series < 7] = 4.0
    weights[hb_series < 10] = 2.0
    weights[hb_series > 15] = 2.0
    return weights

# --------------------------------------------------------------------------
# 2. Main execution logic
# --------------------------------------------------------------------------

def main():
    print("--- Starting External Validation data generation. ---")

    # --- Configuration values ---
    RANDOM_SEED = 42
    
    # Eyedye allocation strategy
    eyedye_allocation = {
        'India': {
            'target': 18,
            'strategy': 'balanced_sampling',  # Uniform across all ranges
            'priority': ['7-10', '10-13', '13-16']
        },
        'Italy': {
            'target': 12, 
            'strategy': 'compensatory_sampling',  # Prioritize insufficient low values
            'priority': ['<9', '9-13', '13-15']  # Minimize extreme high values
        }
    }
    
    # Ghana allocation
    ghana_target = 100  # 14.1% of 708
    
    # Input file paths
    INPUT_CSVS = {
        'eyedye': 'external_validation_joint_results/external_validation_master.csv',
        'ghana': 'external_validation_ghana/anemiaDataGhana.xlsx'
    }
    
    # Create output folder
    output_folder = 'external_validation_proceed'
    os.makedirs(output_folder, exist_ok=True)
    
    # --- Process Eyedye data ---
    print(f"\n[EYEDYE] Processing dataset...")
    df_eyedye = pd.read_csv(INPUT_CSVS['eyedye'])
    
    # Preprocess Hb values
    print("Preprocessing Hb values...")
    # Convert empty strings to NaN
    df_eyedye['Hb'] = df_eyedye['Hb'].replace("", np.nan)
    # Remove rows containing "_"
    df_eyedye = df_eyedye[df_eyedye['Hb'] != "_"]
    # Convert to float
    df_eyedye['Hb'] = pd.to_numeric(df_eyedye['Hb'], errors='coerce')
    # Remove NaN
    df_eyedye = df_eyedye.dropna(subset=['Hb'])
    
    # Fix full_path (exteranal_validation -> external_validation_joint_results)
    # Original data has typo, so it's written as exteranal_validation
    df_eyedye['full_path'] = df_eyedye['full_path'].str.replace(
        'exteranal_validation/', 
        'external_validation_joint_results/', 
        regex=False
    )
    # Additional processing in case normal paths also exist
    df_eyedye['full_path'] = df_eyedye['full_path'].str.replace(
        'external_validation/', 
        'external_validation_joint_results/', 
        regex=False
    )
    
    print(f"Eyedye data after preprocessing: {len(df_eyedye)} samples")
    print(f"India: {len(df_eyedye[df_eyedye['Country'] == 'India'])}")
    print(f"Italy: {len(df_eyedye[df_eyedye['Country'] == 'Italy'])}")
    
    # Separate by country
    df_india = df_eyedye[df_eyedye['Country'] == 'India'].copy()
    df_italy = df_eyedye[df_eyedye['Country'] == 'Italy'].copy()
    
    # Sample by country
    india_train_indices = eyedye_india_balanced_sampling(df_india, eyedye_allocation['India']['target'], RANDOM_SEED)
    italy_train_indices = eyedye_italy_compensatory_sampling(df_italy, eyedye_allocation['Italy']['target'], RANDOM_SEED)
    
    # Combine all train indices
    eyedye_train_indices = india_train_indices + italy_train_indices
    
    # Train/Test split
    df_eyedye_train = df_eyedye.loc[eyedye_train_indices]
    df_eyedye_test = df_eyedye.drop(eyedye_train_indices)
    
    # Save
    eyedye_train_path = os.path.join(output_folder, 'train_ext_eyedye.csv')
    eyedye_test_path = os.path.join(output_folder, 'test_ext_eyedye.csv')
    
    df_eyedye_train.to_csv(eyedye_train_path, index=False)
    df_eyedye_test.to_csv(eyedye_test_path, index=False)
    
    print(f"Eyedye Train Set: {len(df_eyedye_train)} samples ({len(df_eyedye_train)/len(df_eyedye)*100:.1f}%)")
    print(f"  - India: {len(df_eyedye_train[df_eyedye_train['Country'] == 'India'])}")
    print(f"  - Italy: {len(df_eyedye_train[df_eyedye_train['Country'] == 'Italy'])}")
    print(f"Eyedye Test Set: {len(df_eyedye_test)} samples")
    print(f"Saved: {eyedye_train_path}, {eyedye_test_path}")
    
    # --- Process Ghana data ---
    print(f"\n[GHANA] Processing dataset...")
    df_ghana = pd.read_excel(INPUT_CSVS['ghana'])
    
    # Preprocess Hb values
    print("Preprocessing Hb values...")
    # Convert empty strings to NaN
    df_ghana['Hb'] = df_ghana['Hb'].replace("", np.nan)
    # Remove rows containing "_"
    df_ghana = df_ghana[df_ghana['Hb'] != "_"]
    # Convert to float
    df_ghana['Hb'] = pd.to_numeric(df_ghana['Hb'], errors='coerce')
    # Remove NaN
    df_ghana = df_ghana.dropna(subset=['Hb'])
    
    # Add full_path to Ghana data
    img_base_path = 'external_validation_ghana/total_ghana_img'
    df_ghana['full_path'] = df_ghana['w_filename'].apply(lambda x: os.path.join(img_base_path, f"{x}.png"))
    
    print(f"Ghana data after preprocessing: {len(df_ghana)} samples")
    
    # Ghana sampling
    ghana_train_indices = ghana_sampling(df_ghana, ghana_target, RANDOM_SEED)
    
    # Train/Test split
    df_ghana_train = df_ghana.loc[ghana_train_indices]
    df_ghana_test = df_ghana.drop(ghana_train_indices)
    
    # Save
    ghana_train_path = os.path.join(output_folder, 'train_ext_ghana.csv')
    ghana_test_path = os.path.join(output_folder, 'test_ext_ghana.csv')
    
    df_ghana_train.to_csv(ghana_train_path, index=False)
    df_ghana_test.to_csv(ghana_test_path, index=False)
    
    print(f"Ghana Train Set: {len(df_ghana_train)} samples ({len(df_ghana_train)/len(df_ghana)*100:.1f}%)")
    print(f"Ghana Test Set: {len(df_ghana_test)} samples")
    print(f"Saved: {ghana_train_path}, {ghana_test_path}")
    
    # --- Output summary information ---
    print(f"\n--- Generation Summary ---")
    print(f"Eyedye: {len(df_eyedye_train)}/{len(df_eyedye)} = {len(df_eyedye_train)/len(df_eyedye)*100:.1f}%")
    print(f"Ghana: {len(df_ghana_train)}/{len(df_ghana)} = {len(df_ghana_train)/len(df_ghana)*100:.1f}%")
    print(f"All files saved in '{output_folder}' folder.")

if __name__ == '__main__':
    main()