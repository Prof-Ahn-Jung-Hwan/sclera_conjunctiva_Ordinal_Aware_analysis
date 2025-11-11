from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


class AjouMC_AnemiaDataset(Dataset):
    def __init__(self, args, data, mode="train", w_filename=False):
        self.args = args
        self.data = data
        self.mode = mode
        self.w_filename = w_filename

        self.transform = (
            T.Compose(
                [
                    T.Resize((int(args.img_size * 1.143), int(args.img_size * 1.143))),
                    T.RandomCrop(args.img_size),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(30, expand=True),
                    T.Resize((args.img_size, args.img_size)),
                    T.RandomPerspective(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            if mode == "train"
            else T.Compose(
                [
                    T.Resize((args.img_size, args.img_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )

    def __getitem__(self, index):
        img_path, data = self.data[index]
        Hb = data["Hb"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        Hb = min(Hb, self.args.Hb_maxima)
        Hb = torch.tensor([Hb], dtype=torch.float)

        # Return a tuple that matches the expected structure in training scripts
        return (img, Hb, torch.zeros(1), img_path.name) if self.w_filename else (img, Hb, torch.zeros(1))

    def __len__(self):
        return len(self.data)


def load_ajoumc_seghgb_dataset(args, cv_fold=5, seed=42, w_filename=False):
    def _load_anno(root_path):
        """
        Loads annotations from CSV files in the sample_data/annotations directory.
        Combines train and test samples for K-fold cross-validation.
        """
        sample_data_path = root_path / "sample_data"
        train_csv_path = sample_data_path / "annotations/train_sample.csv"
        test_csv_path = sample_data_path / "annotations/test_sample.csv"

        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        df = pd.concat([df_train, df_test], ignore_index=True)

        # Iterate through the dataframe and load data from 'file_path' and 'Hb' columns.
        for _, row in df.iterrows():
            # The 'full_path' in the CSV is relative to the project root.
            file_path = Path(row.get("full_path"))
            hb_val = row.get("Hb")
            if file_path and hb_val is not None:
                data.append((file_path, {"Hb": float(hb_val)}))

    root = Path(args.dataset_root)
    data = []

    _load_anno(root)

    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=seed)

    train_data_list = []
    test_data_list = []
    imbalance_list = []
    img_num_list = []

    for _, (train_index, test_index) in enumerate(kf.split(data)):
        fold_train = [data[i] for i in train_index]
        fold_test = [data[i] for i in test_index]

        train_dataset = AjouMC_AnemiaDataset(args, fold_train, mode="train", w_filename=w_filename)
        test_dataset = AjouMC_AnemiaDataset(args, fold_test, mode="test", w_filename=w_filename)
        imbalance, _N_SAMPLES_PER_CLASS = calc_imbalance(args, fold_train)

        train_data_list.append(train_dataset)
        test_data_list.append(test_dataset)
        imbalance_list.append(imbalance)
        img_num_list.append(_N_SAMPLES_PER_CLASS)

    return train_data_list, test_data_list, imbalance_list, img_num_list


def load_external_validation_dataset(args, cv_fold=5, w_filename=False):
    """
    Loads data for external validation from a single CSV file.
    This function is designed for testing only and does not create training/validation splits.
    It reads a pre-processed CSV which should contain 'full_path', 'Hb', '나이', '성별'.
    """
    # The path to the master CSV file should be provided in the config via 'test_file'.
    if not hasattr(args, "test_file") or not args.test_file:
        raise ValueError(
            "A path to the external validation CSV is required. Please specify it in the config using 'test_file'."
        )

    csv_path = args.test_file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: External validation file not found at '{csv_path}'")
        return [], [], [], []

    # --- Data Cleaning and Preparation ---
    # Handle comma decimal separator in 'Hb' and convert to numeric, dropping invalid rows.
    df["Hb"] = pd.to_numeric(df["Hb"].astype(str).str.replace(",", "."), errors="coerce")
    df.dropna(subset=["Hb"], inplace=True)

    data = []
    for _, row in df.iterrows():
        # The full path is already prepared in the CSV
        img_path = Path(row["full_path"])

        data_dict = {}
        data_dict["Hb"] = float(row["Hb"])

        data.append((img_path, data_dict))

    print(f"✅ Loaded {len(data)} samples for external validation from '{csv_path}'.")

    # Create the test dataset
    test_dataset = AjouMC_AnemiaDataset(args, data, mode="test", w_filename=w_filename)

    # To maintain compatibility with scripts that expect a list of datasets per fold,
    # we replicate the single test dataset for each fold.
    test_data_list = [test_dataset] * cv_fold

    return [], test_data_list, [], []


def calc_imbalance(args, data):
    # Check if this is for the Cross-Entropy experiment
    is_ce_loss = hasattr(args, 'loss_type') and args.loss_type == 'cross_entropy'

    if is_ce_loss:
        # --- Logic for Cross-Entropy Loss ---
        total = len(data) + 1
        imbalance = [1] * (args.ordinal_bins - 1)
        # N_SAMPLES_PER_CLASS must be a histogram of size `ordinal_bins`.
        N_SAMPLES_PER_CLASS = [0] * args.ordinal_bins
        
        for _, anno in data:
            Hb = anno["Hb"]
            shifted_Hb = Hb + args.disc_shift

            # Determine the class index for the sample
            class_idx = 0
            for k in range(args.ordinal_bins - 1):
                if shifted_Hb >= args.lookup[k] and shifted_Hb < args.lookup[k + 1]:
                    class_idx = k
                    break
            if shifted_Hb >= args.lookup[args.ordinal_bins - 1]:
                class_idx = args.ordinal_bins - 1
            
            # Populate histogram for CE
            N_SAMPLES_PER_CLASS[class_idx] += 1
            
            # Populate cumulative count for imbalance_ratio (though not used by CE script)
            for k in range(class_idx + 1):
                if k < len(imbalance):
                    imbalance[k] += 1

        imbalance_ratio = [total / count for count in imbalance]
        return torch.FloatTensor(imbalance_ratio), N_SAMPLES_PER_CLASS

    else:
        # --- Original Logic for Ordinal/IB Loss ---
        total = len(data) + 1
        imbalance = [1] * (args.ordinal_bins - 1)
        N_SAMPLES_PER_CLASS = [0] * (args.ordinal_bins - 1)
        for _, anno in data:
            Hb = anno["Hb"]
            shifted_Hb = Hb + args.disc_shift
            for k in range(args.ordinal_bins - 1):
                imbalance[k] += 1
                N_SAMPLES_PER_CLASS[k] += 1
                if shifted_Hb >= args.lookup[k] and shifted_Hb < args.lookup[k + 1]:
                    break

        imbalance_ratio = []
        for j in range(len(imbalance)):
            imbalance_ratio.append(total / imbalance[j])
        return torch.FloatTensor(imbalance_ratio), N_SAMPLES_PER_CLASS
