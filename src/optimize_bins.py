# optimize_bins.py
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

# Disable Optuna's default log output to avoid interfering with tqdm progress bar.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def bootstrap_ci(data, n_bootstraps=1000, ci=0.95):
    """Calculates the confidence interval using bootstrapping."""
    bootstrapped_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    sorted_means = np.array(bootstrapped_means)
    sorted_means.sort()
    lower_bound = sorted_means[int((1.0 - ci) / 2.0 * len(sorted_means))]
    upper_bound = sorted_means[int((1.0 - (1.0 - ci) / 2.0) * len(sorted_means))]

    return lower_bound, upper_bound


def run_experiment_and_get_mae(config_file, bins_value, summary_csv, gpu_id):
    """
    Calls rerun_single_experiment.sh to run 5-fold CV and
    reads the Mean_MAE value from the result CSV file and returns it.
    """
    base_name = Path(config_file).stem.replace("ajoumc_rxt50_", "")
    exp_name = f"{base_name}-bins{bins_value}"

    # Check if results already exist
    if summary_csv.exists():
        try:
            # Try multiple methods to safely read CSV file
            try:
                df = pd.read_csv(summary_csv, comment="#")
            except pd.errors.ParserError:
                # If parsing error occurs, retry excluding comment lines
                with open(summary_csv, "r") as f:
                    lines = [line for line in f if not line.strip().startswith("#")]

                from io import StringIO

                df = pd.read_csv(StringIO("".join(lines)))

            if exp_name in df["Experiment"].values:
                mae = df[df["Experiment"] == exp_name]["Mean_MAE"].iloc[0]
                # print(f"Found existing result for {exp_name}. MAE: {mae:.4f}") # Commented out to avoid overlap with tqdm
                return mae
        except Exception as e:
            print(f"Warning: Could not read existing results from {summary_csv}: {e}")
            # File may be corrupted, so regenerate it

    # Execute script
    # print(f"Running new experiment for {exp_name} on GPU {gpu_id}...") # Commented out as tqdm shows progress
    # Execute on specified GPU
    command = [
        "./rerun_single_experiment.sh",
        config_file,
        str(bins_value),
        str(summary_csv),
        "",  # No extra_flags
        str(gpu_id),  # GPU ID
    ]

    try:
        # Set capture_output=True to hide child process output and keep main progress bar clean.
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        # Find and display the final summary line output by child script (rerun_single_experiment.sh).
        summary_line = [line for line in result.stdout.split("\n") if "TRIAL_SUMMARY:" in line]
        if summary_line:
            print(summary_line[0])

    except subprocess.CalledProcessError as e:
        # Output detailed information only when an error occurs.
        print(f"\n\nError running experiment for {exp_name} (bins={bins_value}) on GPU {gpu_id}")
        print("--- STDERR ---")
        print(e.stderr)
        print("--- STDOUT ---")
        print(e.stdout)
        return float("inf")

    # Read MAE from result file after execution
    try:
        try:
            df = pd.read_csv(summary_csv, comment="#")
        except pd.errors.ParserError:
            # If parsing error occurs, retry excluding comment lines
            with open(summary_csv, "r") as f:
                lines = [line for line in f if not line.strip().startswith("#")]

            from io import StringIO

            df = pd.read_csv(StringIO("".join(lines)))

        mae = df[df["Experiment"] == exp_name]["Mean_MAE"].iloc[0]
        return mae
    except Exception as e:
        print(f"Error reading results from {summary_csv}: {e}")
        return float("inf")


def objective(trial, config_file, summary_csv, gpu_id):
    """Objective function for Optuna optimization"""
    # Range of bins values to explore. e.g., integers between 10 and 100
    bins = trial.suggest_int("bins", 10, 100)

    # Run experiment and return MAE (MAE is better when lower, so it's a minimization target)
    mae = run_experiment_and_get_mae(config_file, bins, summary_csv, gpu_id)
    return mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize 'bins' hyperparameter using Bayesian Optimization.")
    parser.add_argument(
        "--config", default="configs/ajoumc_rxt50_hybrid_all.yaml", help="Config file for the experiment."
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Number of optimization trials to run.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for the experiments.")
    parser.add_argument("--tqdm-pos", type=int, default=0, help="Position for the tqdm progress bar in the terminal.")
    parser.add_argument("--output-dir", default="bins_optimization", help="Output directory for optimization results.")
    args = parser.parse_args()

    # 1. Save results to specified output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    config_name = Path(args.config).stem
    summary_file = output_dir / f"summary_{config_name}.csv"

    # --- Optuna Study setup (using SQLite DB) ---
    # Use DB to save exploration history and allow resuming if interrupted.
    study_name = f"bins-optimization-{config_name}"
    storage_name = f"sqlite:///{output_dir}/optimization.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # Load if a study with the same name exists.
        direction="minimize",
    )

    # 3. Manually create tqdm progress bar for compatibility with older Optuna versions.
    tqdm_desc = f"Model: {config_name.replace('ajoumc_rxt50_', '')}"

    # Calculate number of completed trials to resume execution.
    n_completed = len(study.trials)
    n_to_run = args.n_trials - n_completed

    if n_to_run > 0:
        with tqdm(
            total=args.n_trials, initial=n_completed, desc=tqdm_desc, position=args.tqdm_pos, leave=False
        ) as pbar:
            # Wrapper function to update progress bar each time objective function is executed
            def objective_with_pbar(trial):
                mae = objective(trial, args.config, summary_file, args.gpu_id)
                pbar.update(1)
                pbar.set_postfix(mae=f"{mae:.4f}")
                return mae

            study.optimize(objective_with_pbar, n_trials=n_to_run, n_jobs=1)

    print("\n==================================================")
    print("Optimization Finished!")
    print(f"Number of trials: {len(study.trials)}")

    if n_to_run > 0:
        print(f"Best trial for {args.config}:")
        best_bins = study.best_params["bins"]
        best_mae = study.best_value
        print(f"  MAE: {best_mae:.4f}")
        print(f"  Best Bins: {best_bins}")
        print(f"Summary saved in: {summary_file}")
        print("==================================================")

        # 3. Append final selected bins value to CSV file
        with open(summary_file, "a") as f:
            f.write("\n# --- Optimization Result ---\n")
            f.write(f"# Best Bins,{best_bins}\n")
            f.write(f"# Best MAE,{best_mae:.4f}\n")
        print(f"Appended best parameters to {summary_file}")
    else:
        print("No new trials run. Loading existing data for visualization.")
        print(f"Summary file: {summary_file}")

    # 4. Generate scatter plot for Hyperparameter Optimization
    # BUG FIX: Use the summary file generated by this optimization run, not a hardcoded old file.
    try:
        try:
            df = pd.read_csv(
                summary_file, comment="#"
            )  # Use comment='#' to ignore summary lines at the end of the file.
        except pd.errors.ParserError:
            # If parsing error occurs, retry excluding comment lines
            with open(summary_file, "r") as f:
                lines = [line for line in f if not line.strip().startswith("#")]

            from io import StringIO

            df = pd.read_csv(StringIO("".join(lines)))

        if not df.empty:
            df["bins"] = df["Experiment"].str.extract(r"bins(\d+)").astype(int)

            # Calculate MAE_95%CI (calculate CI using fold-wise MAE)
            fold_names = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
            df["MAE_CI_lower"] = df.apply(lambda row: bootstrap_ci([row[name] for name in fold_names])[0], axis=1)
            df["MAE_CI_upper"] = df.apply(lambda row: bootstrap_ci([row[name] for name in fold_names])[1], axis=1)

            plt.figure(figsize=(10, 6))
            plt.errorbar(
                df["bins"], df["Mean_MAE"], yerr=df["Std_Dev"], fmt="o", capsize=5, label="Mean MAE ± Std Dev"
            )
            plt.fill_between(df["bins"], df["MAE_CI_lower"], df["MAE_CI_upper"], alpha=0.3, label="95% CI")
            plt.xlabel("Bins")
            plt.ylabel("Mean MAE")
            plt.title("Hyperparameter Optimization: Bins vs Mean MAE")
            plt.legend()
            plt.grid(True)

            output_dir = Path("report_250924")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "spp_table1_full_hyperparameter_search_baysian_opti.png")
            plt.close()

            # Save to Excel
            df.to_excel(output_dir / "spp_table1_full_hyperparameter_search_baysian_opti.xlsx", index=False)
            print(f"Graph and Excel saved to {output_dir}")
        else:
            print("No data available for visualization")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Summary file might be empty or corrupted")
