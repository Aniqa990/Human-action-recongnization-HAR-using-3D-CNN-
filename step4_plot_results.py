import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import platform
import time

# # 1. Load logs
# train_log = 'results_x3d/train.txt'  # or results_mobilenet/train.txt
# val_log = 'results_x3d/val.txt'
# batch_log = 'results_x3d/train_batch.txt'

train_log = 'H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/train'
val_log = 'H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/val'
batch_log = 'H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/train_batch'

# --- Google Drive file existence and open test ---
for f in [train_log, val_log, batch_log]:
    print(f"Testing file: {f}")
    if not os.path.exists(f):
        print(f"ERROR: File not found: {f}")
        print("If this is a Google Drive folder, right-click it in Explorer and select 'Available offline'.")
        exit(1)
    try:
        with open(f, 'r') as testfile:
            print(f"First line of {f}: {testfile.readline().strip()}")
    except Exception as e:
        print(f"ERROR: Could not open {f}: {e}")
        print("If this is a Google Drive folder, right-click it in Explorer and select 'Available offline'.")
        exit(1)


def load_log(path):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in results_mobilenet: {os.listdir('results_mobilenet') if os.path.exists('results_mobilenet') else 'results_mobilenet folder not found!'}")
        exit(1)
    data = pd.read_csv(path, sep='\t')
    return data

train = load_log(train_log)
val = load_log(val_log)

# 2. Plot Loss and Accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train['epoch'], train['loss'], label='Train Loss')
plt.plot(val['epoch'], val['loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epoch')

plt.subplot(1,2,2)
plt.plot(train['epoch'], train['acc'], label='Train Acc')
plt.plot(val['epoch'], val['acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')
plt.tight_layout()
plt.savefig('results_x3d/loss_acc.png')
plt.show()

# 3. Confusion Matrix and Classification Report
# You need to run your test script and save predictions and true labels
# Example: save as test_results.json with {'y_true': [...], 'y_pred': [...]}
with open('test_results.json') as f:
    results = json.load(f)
y_true = results['y_true']
y_pred = results['y_pred']
labels = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('results_x3d/confusion_matrix.png')
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# 4. Per-class accuracy bar plot
acc_per_class = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(8,4))
sns.barplot(x=labels, y=acc_per_class)
plt.ylabel('Accuracy')
plt.title('Per-class Accuracy')
plt.savefig('results_x3d/per_class_accuracy.png')
plt.show()

# 5. Hardware and timing info
print("System:", platform.platform())
print("Python:", platform.python_version())
if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
else:
    print("CPU only")

# 6. Model parameter count (example for X3D)
# from step3_train_x3d.py, after model is built:
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params}")

# 7. (Optional) ROC curves, if you want

# 8. (Optional) Plot learning rate schedule, if you logged it

# 9. (Optional) Plot batch-wise loss/accuracy using batch_log

# Save all plots in your results folder for easy inclusion in your paper.

# After loading your model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the output
y_true_bin = label_binarize(y_true, classes=range(len(labels)))
y_pred_prob = ... # shape: (n_samples, n_classes), get from your model's softmax output

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(labels)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend()
plt.savefig('results_x3d/roc_curve.png')
plt.show()

# If you logged learning rate in your train.txt/val.txt
plt.figure()
plt.plot(train['epoch'], train['lr'], label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.savefig('results_x3d/lr_schedule.png')
plt.show()

batch = pd.read_csv(batch_log, sep='\t')
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(batch['iter'], batch['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Batch-wise Loss')

plt.subplot(1,2,2)
plt.plot(batch['iter'], batch['acc'])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Batch-wise Accuracy')
plt.tight_layout()
plt.savefig('results_x3d/batch_loss_acc.png')
plt.show()

summary = {
    "total_params": int(total_params),
    "best_val_acc": float(val['acc'].max()),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True),
    # Add more as needed
}
with open('results_x3d/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)