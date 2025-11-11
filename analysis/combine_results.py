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
            # 데이터가 어떤 fold에서 왔는지 추적하기 위해 'fold' 열을 추가합니다.
            # 디렉토리 이름에서 fold 번호를 추출합니다. (e.g., eyedye_few_fold0 -> 0)
            folder_name = Path(file_path).parent.name
            if "fold" in folder_name:
                # fold 뒤의 숫자를 추출
                fold_num = folder_name.split("fold")[-1]
                try:
                    df["fold"] = int(fold_num)
                except ValueError:
                    # fold 번호 추출 실패시 파일 순서로 대체
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

    # 모든 데이터프레임을 하나로 합칩니다.
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 최종 결과를 새로운 Excel 파일로 저장합니다.
    combined_df.to_excel(args.output, index=False)

    print(f"\nSuccessfully combined all results into '{args.output}'")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()