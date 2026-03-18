# Cross-Modal Visuo Tactile Object Perception

This repository supports research on **cross-modal perception** combining visual and tactile information for understanding object properties. It provides:

- Preprocessing and preparation scripts for cross-modal datasets
- Three model architectures for fusion: **Baseline (SVAE)**, **Joint Latent Filtering**, and **Cross-Modal Latent Filtering (CMLF)**
- Comprehensive evaluation scripts for regression, classification, distance metrics, and dimensionality analysis

## 📦 Dataset Access

Cross-modal perception dataset combining synchronized visual and tactile observations:

This dataset contains:
- Visual observations (images, 128×128, 2 channels)
- Tactile observations (pressure maps, 80×80, 1 channel)
- Action sequences (9-dimensional actions, 99-step horizon)
- Object labels and ground-truth properties

📁 Place the downloaded files in the `dataset/` directory structure before running experiments.

---

## 📁 Directory Structure

```
VT_Cross_Modal_Perception/
├── dataset/
│   ├── cm_dataset/
│   │   ├── raw/                    # Raw unprocessed data
│   │   └── processed/              # Preprocessed data
│   └── gt_objects/                 # 3D CAD of all the objects used in the experiments
├── results/                        # Output of trained models and evaluations
│   ├── classification/             # Classification arrays
│   ├── regression/                 # Regression results
│   │   ├── baseline/
│   │   ├── joint/
│   │   └── w-cm/ (CMLF)
│   ├── euc_distances/              # Distance metrics
│   └── umaps/                      # UMAP visualizations
├── utils/                          # Model implementation and supporting files
│   ├── networks.py                 # DL networks
│   ├── baseline_svae.py            # Baseline SVAE model
│   ├── cross_modal_latent_filter.py  # CMLF model 
│   ├── joint_latent_filter.py      # Joint model
│   ├── datasets.py                 # Data loading utilities
│   ├── compute_utils.py            # Training utilities
│   ├── functions.py                # Loss and activation functions
│   ├── flows.py                    # Normalizing flows
│   ├── dist.py                     # Distribution classes
│   └── plot_latent.py              # Plotting functions during training
├── a_preprocess_data.py            # Data preprocessing
├── b_prepare_data.py               # Data preparation & splitting
├── c_train_baseline.py             # Train baseline SVAE
├── c_train_joint.py                # Train joint latent filter
├── c_train_cmlf.py                 # Train cross-modal latent filter
├── d_test_baseline.py              # Test baseline model
├── d_test_joint.py                 # Test joint model
├── d_test_cmlf.py                  # Test CMLF model
├── e_eval_regression.py            # Evaluate regression performance
├── e_eval_classification.py        # Evaluate classification metrics
├── e_eval_distances.py             # Evaluate latent distances
├── e_eval_UMAP.py                  # Evaluate latent UMAP visualizations
├── f_plot_result_I.py              # Plotting script for results Fig. 4 
├── f_plot_result_II.py             # Plotting script for results Table. 1
├── f_plot_result_III.py            # Plotting script for results Fig. 6
├── f_plot_appendix_res_I.py        # Plotting script for appendix Fig. 3
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

Tested on Python == 3.10, Ubuntu 20.04

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Code Execution Steps (a_**.py ➡️ e_**.py)

```bash
# Step 1: Preprocess dataset (cleans and downsample)
python a_preprocess_data.py

# Step 2: Prepare training/validation/test splits
python b_prepare_data.py

# Step 3a: Train the baseline SVAE model
python c_train_baseline.py --validate True --visdom False

# Step 3b: Train the joint latent filter model
python c_train_joint.py --validate True --visdom False

# Step 3c: Train the cross-modal latent filter (CMLF) model
python c_train_cmlf.py --validate True --visdom False

# Step 4a: Evaluate baseline model on test set
python d_test_baseline.py

# Step 4b: Evaluate joint model on test set
python d_test_joint.py

# Step 4c: Evaluate CMLF model on test set
python d_test_cmlf.py

# Step 5: Run regression evaluation (required before plotting)
python e_eval_regression.py

# Step 6: Run classification evaluation
python e_eval_classification.py

# Step 7: Compute latent space distances
python e_eval_distances.py

# Step 8: Generate UMAP visualizations
python e_eval_UMAP.py

# Result plots corresponding to the manuscript, results matrix present in the repository and can be validated in isolation
python f_plot_result_I.py
python f_plot_result_II.py
python f_plot_result_III.py
python f_plot_appendix_res_I.py
```

> **Optional**: Set `--visdom True` in training scripts if you want real-time training visualization (requires visdom server running)

---
## 📧 Contact & Citation

For questions or dataset access requests, please contact:

📬 **anirvan.dutta95@gmail.com**

---
