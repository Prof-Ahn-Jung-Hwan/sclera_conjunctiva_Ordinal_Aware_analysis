# Non-invasive Hemoglobin Estimation from Conjunctival Images

This repository contains the official PyTorch implementation for the paper "[Cross-Population Non-Invasive Hemoglobin Screening via Ordinal-Aware Sclera-Conjunctiva Co-Analysis]". Our work proposes a deep learning model, HbNet, for non-invasively estimating hemoglobin (Hb) levels from digital images of the conjunctiva.

## Overview

Anemia is a global health problem, and its diagnosis often requires invasive blood tests. This project explores a non-invasive, cost-effective alternative using smartphone-captured images of the eye. The model leverages ordinal regression and a variational encoder to accurately predict Hb levels, providing a valuable tool for point-of-care screening.

Our framework is clinically-driven, inspired by how clinicians reference the white sclera to assess the redness of the conjunctiva. We translated this concept into an engineering solution: using the sclera for intrinsic color standardization. The architecture combines ResNeXt + Dual-Attention for feature extraction, a Variational Encoder (VE) with MMD regularization to mitigate lighting variations and distinguish Hb levels in the latent space, and an ArcFace Ordinal Head with logarithmic binning for the final estimation.

A key strength of this work is its validation. Beyond internal data, we performed few-shot learning on public datasets (from India/Italy) with different characteristics, demonstrating its potential as a robust screening tool. The resulting MAE of 1.1454 is highly competitive, ranking among the lowest in recent literature.

### Key Features
- **HbNet Architecture**: A novel deep learning model for Hb estimation.
- **Ordinal Regression**: Treats Hb estimation as an ordered classification problem, improving accuracy.
- **Variational Encoder**: Helps create a more robust and generalizable feature representation.
- **Hyperparameter Optimization**: Includes scripts for Bayesian optimization to find the optimal number of ordinal bins.
- **Ablation & External Validation**: Scripts to reproduce ablation studies and evaluate model performance on external datasets.

## Project Structure

```
mycode_github_251109/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── configs/                # YAML configuration files for experiments
├── dataset/                # Dataloader and data processing logic
├── model/                  # Model architecture (HbNet, loss functions, etc.)
├── ordinal_regression/     # Ordinal regression helper functions
├── src/                    # Main Python scripts for training, testing, etc.
├── analysis/               # Scripts for analyzing results
├── scripts/                # Shell scripts to run experiments
├── external_validation_proceed/ # Scripts and data for external validation
└── sample_data/            # Sample dataset for demonstration
```

## Setup and Installation

### Prerequisites
- Git
- Anaconda or Miniconda
- NVIDIA GPU with CUDA support

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n anemia-proj python=3.10
    conda activate anemia-proj
    ```

3.  **Install PyTorch:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command for your CUDA version. For example:
    ```bash
    # Example for CUDA 12.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```

4.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This repository includes a small sample dataset derived from the **Eyes-Defy-Anemia** public dataset to demonstrate the code's functionality.

### Sample Data
- **Source**: [Eyes-Defy-Anemia Dataset](https://doi.org/10.21227/t5s2-4j73)
- **Contents**: A small subset of images and annotations from the original dataset.
- **Location**: `sample_data/`
- **Purpose**: For code testing, debugging, and verifying the data pipeline. **Not for performance evaluation.**

### Full Dataset and Citation

To reproduce the results from our paper, you need access to the full internal and external datasets. The provided sample data is from the following public dataset. If you use this data, please cite the original work:

```bibtex
@article{dimauro2023intelligent,
  title={An intelligent non-invasive system for automated diagnosis of anemia exploiting a novel dataset},
  author={Dimauro, Giovanni and others},
  journal={Artificial Intelligence in Medicine},
  volume={136},
  pages={102477},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.artmed.2022.102477}
}
```

## How to Run Experiments

All experiments are executed via shell scripts located in the `scripts/` directory.

**Important Note on Reproducibility**: The following scripts are fully functional and demonstrate the complete workflow of the project using the provided `sample_data`. However, due to the private nature of the internal clinical data, the performance metrics (e.g., MAE) obtained from these scripts will not match the results reported in the paper. The primary purpose is to validate the code's integrity and illustrate the experimental methodology.

### 1. Hyperparameter Optimization (Finding Best Bins)

This script uses Bayesian optimization to find the optimal number of `bins` for the ordinal regression model using the sample data.

```bash
# Run optimization for the image-only model
./scripts/run_bins_optimization_only_image.sh
```
The results, including the best `bins` value, will be saved in the `bins_optimization_only_image/` directory.

### 2. Training a Model from Scratch

To train a model with a specific number of bins (e.g., 60) using 5-fold cross-validation on the sample data, use the `rerun_single_experiment.sh` script.

```bash
# Usage: ./rerun_single_experiment.sh <config_file> <bins> <summary_csv> [extra_flags] [gpu_id]
./scripts/rerun_single_experiment.sh configs/ajoumc_rxt50_image.yaml 60 temp_summary.csv "" 0
```
Training logs and model checkpoints will be saved under `log_bins/train/`.

### 3. Ablation Studies

After finding the best `bins` from optimization, you can run ablation studies (e.g., without the encoder or without ordinal regression).

**Note**: This script demonstrates the process of running ablation studies. The resulting performance will differ from the paper's results due to the use of sample data.

```bash
# This script uses the best bins found previously to run various ablation models.
./scripts/run_ablation_image_bestBins.sh
```

### 4. External Validation (Zero-shot & Few-shot)

These scripts demonstrate how to evaluate a pre-trained model on an "external" dataset.

**Note**: These scripts evaluate a pre-trained model (`image-bins76`) on the provided `sample_data`. They serve to demonstrate the evaluation workflow, not to reproduce the paper's cross-dataset generalization results, which require the original private training data.

#### Zero-Shot Evaluation

Evaluates the performance of the pre-trained model on the sample data without any fine-tuning.

```bash
./scripts/run_zero_shot_image_evaluation.sh
```

#### Few-Shot Evaluation

First fine-tunes the pre-trained model on a small portion of the sample data (simulating a few-shot scenario) and then evaluates its performance.

```bash
./scripts/run_few_shot_image_evaluation.sh
```

## Collaboration on Internal Data

Our full research was conducted using a larger, internal clinical dataset from Ajou University Medical Center. Due to patient privacy and institutional regulations (IRB), this data cannot be made public.

We welcome academic and research collaboration. If you are interested in working with our internal dataset, please adhere to the following protocol:

- **Contact:** erdrajh@ajou.ac.kr
- **Requirements:**
  - IRB approval from your institution and Ajou University Medical Center.
  - A formal Data Use Agreement (DUA).
  - A mutually agreed-upon research collaboration protocol.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. The sample data is subject to the terms of use of the original Eyes-Defy-Anemia dataset.