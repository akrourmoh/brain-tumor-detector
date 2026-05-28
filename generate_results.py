import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

os.makedirs('results', exist_ok=True)

print("Loading model and test data...")
model  = load_model('models/best_model.h5')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No Tumor', 'Tumor'],
    yticklabels=['No Tumor', 'Tumor'],
    annot_kws={'size': 16}
)
ax.set_xlabel('Predicted label', fontsize=13)
ax.set_ylabel('True label', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/confusion_matrix.png")

# --- Sample predictions ---
tumor_correct   = np.where((y_test == 1) & (y_pred == 1))[0]
no_tumor_correct = np.where((y_test == 0) & (y_pred == 0))[0]
false_positive  = np.where((y_test == 0) & (y_pred == 1))[0]
false_negative  = np.where((y_test == 1) & (y_pred == 0))[0]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('Sample Predictions', fontsize=15, fontweight='bold', y=1.01)

samples = [
    (tumor_correct[:2],    'Tumor — Correct',      '#1AA260'),
    (no_tumor_correct[:2], 'No Tumor — Correct',   '#1AA260'),
    (false_positive[:1],   'False Positive',        '#D63232'),
    (false_negative[:1],   'False Negative',        '#D63232'),
]

col = 0
for indices, label, color in samples:
    for idx in indices:
        if col >= 8:
            break
        row = col // 4
        c   = col % 4
        axes[row][c].imshow(X_test[idx])
        prob = float(y_pred_proba[idx][0])
        axes[row][c].set_title(
            f'{label}\nP(tumor) = {prob:.2f}',
            color=color, fontsize=9, fontweight='bold'
        )
        axes[row][c].axis('off')
        col += 1

for i in range(col, 8):
    axes[i // 4][i % 4].axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/sample_predictions.png")

# --- Metrics bar chart ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values  = [acc, precision, recall, f1]
colors  = ['#2D6AF6', '#1AA260', '#F6A02D', '#D63232']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='white')
ax.set_ylim(0.75, 1.0)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance on Test Set (281 images)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results/metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/metrics.png")

print(f"\nDone. Accuracy={acc:.3f} | F1={f1:.3f} | Precision={precision:.3f} | Recall={recall:.3f}")
