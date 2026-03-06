"""
This code evaluates the LR and SVM classification outputs
"""
import os
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)  # For numpy-bas

baseline_test = np.load(os.path.join('results', 'baseline', 'model_output', 'out_test_n0_c0.npz'))
baseline_val = np.load(os.path.join('results', 'baseline', 'model_output', 'out_val_n0_c0.npz'))
joint_test = np.load(os.path.join('results', 'joint', 'model_output', 'out_test_n0_c0.npz'))
joint_val = np.load(os.path.join('results', 'joint', 'model_output', 'out_val_n0_c0.npz'))
wcm_test = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_test_n0_c0.npz'))
wcm_val = np.load(os.path.join('results', 'w-cm', 'late', 'model_output', 'out_val_n0_c0.npz'))
wocm_test = np.load(os.path.join('results', 'wo-cm', 'model_output', 'out_test_n0_c0.npz'))
wocm_val = np.load(os.path.join('results', 'wo-cm', 'model_output', 'out_val_n0_c0.npz'))

# Merge test and val for baseline
baseline_y_params = np.concatenate([baseline_test['y_params'], baseline_val['y_params']])
baseline_labels_test = np.concatenate([baseline_test['labels_test'], baseline_val['labels_test']])
baseline_labels = baseline_labels_test[:, -1, :]
baseline_labels_np = ((baseline_labels[:, 0] - 1) * 15 + (baseline_labels[:, 1] - 1) * 5 + (baseline_labels[:, 2] - 1))
baseline_features_np = baseline_y_params[:, -1, :, 0]

# Merge test and val for joint
joint_y_params = np.concatenate([joint_test['y_params'], joint_val['y_params']])
joint_labels_test = np.concatenate([joint_test['labels_test'], joint_val['labels_test']])
joint_labels = joint_labels_test[:, -1, :]
joint_labels_np = ((joint_labels[:, 0] - 1) * 15 + (joint_labels[:, 1] - 1) * 5 + (joint_labels[:, 2] - 1))
joint_features_np = joint_y_params[:, -1, :, 0]

# Merge test and val for wcm (with cross-modal)
wcm_vis_y_params = np.concatenate([wcm_test['vis_y_params'], wcm_val['vis_y_params']])
wcm_tac_y_params = np.concatenate([wcm_test['tac_y_params'], wcm_val['tac_y_params']])
wcm_labels_test = np.concatenate([wcm_test['labels_test'], wcm_val['labels_test']])
wcm_labels = wcm_labels_test[:, -1, :]
wcm_labels_np = ((wcm_labels[:, 0] - 1) * 15 + (wcm_labels[:, 1] - 1) * 5 + (wcm_labels[:, 2] - 1))
wcm_vis_features_np = wcm_vis_y_params[:, -1, :, 0]
wcm_tac_features_np = wcm_tac_y_params[:, -1, :, 0]
wcm_features_np = np.concatenate([wcm_vis_features_np, wcm_tac_features_np], axis=1)

# Merge test and val for wocm (without cross-modal)
wocm_vis_y_params = np.concatenate([wocm_test['vis_y_params'], wocm_val['vis_y_params']])
wocm_tac_y_params = np.concatenate([wocm_test['tac_y_params'], wocm_val['tac_y_params']])
wocm_labels_test = np.concatenate([wocm_test['labels_test'], wocm_val['labels_test']])
wocm_labels = wocm_labels_test[:, -1, :]
wocm_labels_np = ((wocm_labels[:, 0] - 1) * 15 + (wocm_labels[:, 1] - 1) * 5 + (wocm_labels[:, 2] - 1))
wocm_vis_features_np = wocm_vis_y_params[:, -1, :, 0]
wocm_tac_features_np = wocm_tac_y_params[:, -1, :, 0]
wocm_features_np = np.concatenate([wocm_vis_features_np, wocm_tac_features_np], axis=1)


feature_sets = {"baseline": baseline_features_np,
                "joint": joint_features_np,
                "wocm_vision": wocm_vis_features_np,
                "wocm_tactile": wocm_tac_features_np,
                "wocm_joint": wocm_features_np,
                "wcm_vision": wcm_vis_features_np,
                "wcm_tactile": wcm_tac_features_np,
                "wcm_joint": wcm_features_np}

# Use WCM labels (they are the same across feature sets)
object_labels = wcm_labels_np

# Store average accuracies
results = {
    "Feature Set": [],
    "all LR": [],
    "LR mean": [],
    "LR std": [],
    "all SVM": [],
    "SVM mean": [],
    "SVM std": [],
}

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=0)

for name, features in feature_sets.items():
    lr_fold_accuracies = []
    svm_fold_accuracies = []
    print(f"\n{name} - Stratified Cross-Validation:")

    for train_index, test_index in rskf.split(features, object_labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = object_labels[train_index], object_labels[test_index]

        # Scale
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic Regression
        lr_model = LogisticRegression(random_state=0, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)
        lr_fold_accuracies.append(balanced_accuracy_score(y_test, lr_preds))

        # SVM
        svm_model = SVC(kernel='rbf', gamma='scale', random_state=0)
        svm_model.fit(X_train, y_train)
        svm_preds = svm_model.predict(X_test)
        svm_fold_accuracies.append(balanced_accuracy_score(y_test, svm_preds))

    # Convert to numpy arrays
    lr_fold_accuracies = np.array(lr_fold_accuracies)
    svm_fold_accuracies = np.array(svm_fold_accuracies)

    avg_lr = np.mean(lr_fold_accuracies)
    avg_svm = np.mean(svm_fold_accuracies)
    std_lr = np.std(lr_fold_accuracies)
    std_svm = np.std(svm_fold_accuracies)

    print(f"  Avg LR Accuracy: {avg_lr:.4f} ± {std_lr:.4f}")
    print(f"  Avg SVM Accuracy: {avg_svm:.4f} ± {std_svm:.4f}")

    results["Feature Set"].append(name)
    results["all LR"].append(lr_fold_accuracies)
    results["LR mean"].append(avg_lr)
    results["LR std"].append(std_lr)
    results["all SVM"].append(svm_fold_accuracies)
    results["SVM mean"].append(avg_svm)
    results["SVM std"].append(std_svm)

# Convert to arrays
feature_names = np.array(results["Feature Set"])
logreg_fold_accuracies = np.array(results["all LR"])  # Shape: (n_feature_sets, n_folds)
logreg_accuracies = np.array(results["LR mean"])
logreg_stds = np.array(results["LR std"])
svm_fold_accuracies = np.array(results["all SVM"])  # Shape: (n_feature_sets, n_folds)
svm_accuracies = np.array(results["SVM mean"])
svm_stds = np.array(results["SVM std"])

# Create output directory if it doesn't exist
os.makedirs(os.path.join('results', 'classification'), exist_ok=True)

# Save as .npy files
np.save(os.path.join('results', 'classification', 'feature_names.npy'), feature_names)
np.save(os.path.join('results', 'classification', 'logreg_accuracies.npy'), logreg_accuracies)
np.save(os.path.join('results', 'classification', 'logreg_stds.npy'), logreg_stds)
np.save(os.path.join('results', 'classification', 'svm_accuracies.npy'), svm_accuracies)
np.save(os.path.join('results', 'classification', 'svm_stds.npy'), svm_stds)

# Save individual fold accuracies for paired t-tests
np.save(os.path.join('results', 'classification', 'logreg_fold_accuracies.npy'), logreg_fold_accuracies)
np.save(os.path.join('results', 'classification', 'svm_fold_accuracies.npy'), svm_fold_accuracies)

'''
# Quick Plotting
import matplotlib.pyplot as plt
x = np.arange(len(results["Feature Set"]))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, results["Logistic Regression"], width, label='Logistic Regression')
plt.bar(x + width/2, results["SVM"], width, label='SVM')
plt.xticks(x, results["Feature Set"], rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy by Feature Type')
plt.legend()
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()
'''


