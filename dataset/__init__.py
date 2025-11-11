from dataset.ajoumc_anemia import (
    load_ajoumc_seghgb_dataset,
    load_external_validation_dataset,
)


def load_dataset(args, cv_fold=5, seed=42, w_filename=False):
    dataset_name = args.dataset.lower()
    if dataset_name.startswith("ajoumc") or dataset_name == "sample_data":
        return load_ajoumc_seghgb_dataset(args, cv_fold, seed, w_filename)
    elif dataset_name == "external":
        return load_external_validation_dataset(args, cv_fold, w_filename)
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported!")
