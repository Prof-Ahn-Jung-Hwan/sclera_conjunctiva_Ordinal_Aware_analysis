# combine_results.py
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple results.xlsx files from k-fold cross-validation."
    )
    parser.add_argument(
        "file_paths", nargs="+", help="List of paths to the results.xlsx files."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results_all_folds_combined.xlsx",
        help="Name of the output combined excel file.",
    )
    args = parser.parse_args()

    print(f"Combining {len(args.file_paths)} files:")
    for path in args.file_paths:
        print(f" - {path}")

    all_dfs = []
    for file_path in args.file_paths:
        try:
            df = pd.read_excel(file_path)
            # Add 'fold' column to track which fold the data came from.
            # Extract fold number from directory name. (e.g., eyedye_few_fold0 -> 0)
            folder_name = Path(file_path).parent.name
            if "fold" in folder_name:
                # Extract number after fold
                fold_num = folder_name.split("fold")[-1]
                try:
                    df["fold"] = int(fold_num)
                except ValueError:
                    # If fold number extraction fails, use file order as replacement
                    df["fold"] = len(all_dfs)
                    print(f"Warning: Could not extract fold number from {folder_name}, using index {len(all_dfs)}")
            else:
                df["fold"] = len(all_dfs)
                print(f"Warning: No fold pattern found in {folder_name}, using index {len(all_dfs)}")
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Error: {e}")

    if not all_dfs:
        print("No valid files to combine.")
        return

    # Combine all DataFrames into one
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save final results to new Excel file
    combined_df.to_excel(args.output, index=False)

    print(f"\nSuccessfully combined all results into '{args.output}'")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()