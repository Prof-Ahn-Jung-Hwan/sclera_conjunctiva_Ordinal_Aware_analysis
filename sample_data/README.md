# Sample Data for Hemoglobin Estimation

## Overview
This directory contains sample data from the **Eyes-Defy-Anemia** dataset for demonstration and testing purposes.

## Dataset Attribution

### Original Dataset
**Title:** Eyes-Defy-Anemia Dataset  
**Authors:** Giovanni Dimauro, et al.  
**Publication:** "An intelligent non-invasive system for automated diagnosis of anemia exploiting a novel dataset"  
**Journal:** Artificial Intelligence in Medicine, Volume 136 (2023) 102477  
**DOI:** https://doi.org/10.1016/j.artmed.2022.102477  
**Dataset DOI:** https://dx.doi.org/10.21227/t5s2-4j73  
**License:** CC BY-NC-ND 4.0 (for the paper), Dataset available on IEEE DataPort  

### Data Access
- **Full Dataset:** Available at [IEEE DataPort](https://ieee-dataport.org/documents/eyes-defy-anemia)
- **Sample Size:** 13 images (8 training, 5 testing) from India and Italy cohorts
- **Purpose:** Code demonstration and quick testing only

## Citation Requirements

If you use this sample data or the full Eyes-Defy-Anemia dataset, please cite:

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

## Directory Structure

```
sample_data/
├── README.md                    # This file
├── images/
│   ├── india/                   # Indian cohort samples
│   │   ├── i_*.jpg             # 8 images
│   └── italy/                   # Italian cohort samples
│       └── *.jpg               # 5 images
└── annotations/
    ├── train_sample.csv        # Training set annotations (8 samples)
    └── test_sample.csv         # Test set annotations (5 samples)
```

## Annotation Format

CSV files contain the following columns:
- `full_path`: Relative path to the image file
- `w_filename`: Image filename without extension
- `Hb`: Hemoglobin level (g/dL) - Ground truth label
- `age`: Patient age (years)
- `sex`: Patient sex (1.0 = Male, 0.0 = Female)
- `Country`: Data source country (India or Italy)

## Usage

These sample images can be used to:
1. Test the code installation
2. Verify the data loading pipeline
3. Run quick experiments for debugging
4. Understand the expected data format

## Limitations

- **Not for clinical use:** Sample size too small for meaningful analysis
- **Demonstration only:** For full experiments, request access to complete dataset
- **IRB required:** For using internal clinical data, institutional IRB approval needed

## Full Dataset Access

To access the complete Eyes-Defy-Anemia dataset:
1. Visit [IEEE DataPort](https://ieee-dataport.org/documents/eyes-defy-anemia)
2. Create a free account
3. Download the full dataset
4. Cite the original paper

## Internal Data Collaboration

For collaboration using internal Ajou Medical Center data:
- **Contact:** erdrajh@ajou.ac.kr
- **Requirements:** 
  - IRB approval from your institution and my institution
  - Data Use Agreement (DUA)
  - Research collaboration protocol

## License

The sample images are from the Eyes-Defy-Anemia dataset and subject to its original terms of use. The code in this repository is licensed under the MIT License (see LICENSE file in the root directory).
