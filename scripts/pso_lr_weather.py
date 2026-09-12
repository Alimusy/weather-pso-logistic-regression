# ============================================================
# PSO Feature Selection + Logistic Regression
# Weather Prediction (Rain in Australia Dataset)
# ============================================================


# ── CELL 1: Install Dependencies ─────────────────────────────
# (Colab magic, run this in a shell instead) !pip install pyswarms imbalanced-learn scikit-learn pandas numpy matplotlib seaborn joblib -q


# ── CELL 2: Imports ──────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pyswarms as ps
from pyswarms.discrete import BinaryPSO

warnings.filterwarnings("ignore")
np.random.seed(42)
print("✅ All libraries loaded.")


# ── CELL 3: Load Dataset ─────────────────────────────────────
# Upload weatherAUS.csv manually in Colab, or mount Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/MyDrive/weatherAUS.csv')

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

print(f"Dataset shape: {df.shape}")
print(df.head(3))


# ── CELL 4: Preprocessing ────────────────────────────────────
# 4a. Drop rows where target is missing
df.dropna(subset=['RainTomorrow'], inplace=True)

# 4b. Cyclic month encoding from Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
df.drop(columns=['Date'], inplace=True)

# 4c. Drop Location (too many categories; not useful without encoding strategy)
df.drop(columns=['Location'], inplace=True)

# 4d. Encode binary categoricals
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# 4e. Encode wind direction columns
wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
le = LabelEncoder()
for col in wind_cols:
    df[col] = df[col].fillna('Unknown')
    df[col] = le.fit_transform(df[col].astype(str))

# 4f. Fill remaining numeric nulls with median
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print(f"After preprocessing: {df.shape}")
print(f"Class distribution:\n{df['RainTomorrow'].value_counts()}")


# ── CELL 5: Feature/Target Split + Scaling ───────────────────
X = df.drop(columns=['RainTomorrow']).values
y = df['RainTomorrow'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

feature_names = df.drop(columns=['RainTomorrow']).columns.tolist()
print(f"Total features: {len(feature_names)}")
print(f"Features: {feature_names}")


# ── CELL 6: Baseline LR (All Features, 5-Fold CV) ───────────
print("\n⏳ Running Baseline LR (all features)...")

lr_base = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', random_state=42)

pipeline_base = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', lr_base)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Collect metrics across folds
base_metrics = {'acc': [], 'f1': [], 'prec': [], 'rec': [], 'auc': []}

for train_idx, test_idx in skf.split(X_scaled, y):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipeline_base.fit(X_tr, y_tr)
    y_pred = pipeline_base.predict(X_te)
    y_prob = pipeline_base.predict_proba(X_te)[:, 1]

    base_metrics['acc'].append(accuracy_score(y_te, y_pred))
    base_metrics['f1'].append(f1_score(y_te, y_pred))
    base_metrics['prec'].append(precision_score(y_te, y_pred))
    base_metrics['rec'].append(recall_score(y_te, y_pred))
    base_metrics['auc'].append(roc_auc_score(y_te, y_prob))

print("\n📊 Baseline LR Results (Mean ± Std across 5 folds):")
for k, v in base_metrics.items():
    print(f"  {k.upper():<6}: {np.mean(v):.4f} ± {np.std(v):.4f}")


# ── CELL 7: PSO Feature Selection ────────────────────────────
# Use a stratified subsample for PSO fitness (speed)
SAMPLE_SIZE = 20000

idx_sample = []
for cls in [0, 1]:
    cls_idx = np.where(y == cls)[0]
    n = int(SAMPLE_SIZE * (len(cls_idx) / len(y)))
    idx_sample.extend(np.random.choice(cls_idx, n, replace=False))

X_sample = X_scaled[idx_sample]
y_sample = y[idx_sample]

print(f"PSO fitness sample size: {len(X_sample)}")
print(f"Sample class distribution: {np.bincount(y_sample)}")

# Fitness function: minimise negative F1 (with SMOTE + LR inside)
def pso_fitness(particles):
    """
    particles: (n_particles, n_features) binary matrix
    returns: (n_particles,) cost array
    """
    costs = []
    for particle in particles:
        selected = np.where(particle == 1)[0]

        # Ensure at least 3 features selected
        if len(selected) < 3:
            costs.append(1.0)  # worst cost
            continue

        X_sel = X_sample[:, selected]

        lr = LogisticRegression(max_iter=300, class_weight='balanced',
                                solver='saga', random_state=42)
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('lr', lr)
        ])

        skf_pso = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []

        for tr, te in skf_pso.split(X_sel, y_sample):
            pipeline.fit(X_sel[tr], y_sample[tr])
            y_pred = pipeline.predict(X_sel[te])
            f1_scores.append(f1_score(y_sample[te], y_pred, zero_division=0))

        costs.append(1 - np.mean(f1_scores))  # minimise

    return np.array(costs)


# PSO parameters
N_PARTICLES = 30
N_ITERATIONS = 100
N_FEATURES = X_scaled.shape[1]

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

print(f"\n⏳ Running Binary PSO ({N_PARTICLES} particles, {N_ITERATIONS} iterations)...")
print("This may take several minutes — grab water 💧\n")

optimizer = BinaryPSO(n_particles=N_PARTICLES,
                      dimensions=N_FEATURES,
                      options=options)

cost_history, best_pos = optimizer.optimize(pso_fitness, iters=N_ITERATIONS, verbose=True)

selected_features = np.where(best_pos == 1)[0]
selected_names = [feature_names[i] for i in selected_features]

print(f"\n✅ PSO done!")
print(f"Selected {len(selected_features)} / {N_FEATURES} features:")
print(selected_names)


# ── CELL 8: PSO+LR Training & Evaluation (5-Fold CV) ─────────
print("\n⏳ Running PSO+LR on selected features (5-fold CV)...")

X_pso = X_scaled[:, selected_features]

lr_pso = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', random_state=42)

pipeline_pso = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', lr_pso)
])

pso_metrics = {'acc': [], 'f1': [], 'prec': [], 'rec': [], 'auc': []}

for train_idx, test_idx in skf.split(X_pso, y):
    X_tr, X_te = X_pso[train_idx], X_pso[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipeline_pso.fit(X_tr, y_tr)
    y_pred = pipeline_pso.predict(X_te)
    y_prob = pipeline_pso.predict_proba(X_te)[:, 1]

    pso_metrics['acc'].append(accuracy_score(y_te, y_pred))
    pso_metrics['f1'].append(f1_score(y_te, y_pred))
    pso_metrics['prec'].append(precision_score(y_te, y_pred))
    pso_metrics['rec'].append(recall_score(y_te, y_pred))
    pso_metrics['auc'].append(roc_auc_score(y_te, y_prob))

print("\n📊 PSO+LR Results (Mean ± Std across 5 folds):")
for k, v in pso_metrics.items():
    print(f"  {k.upper():<6}: {np.mean(v):.4f} ± {np.std(v):.4f}")


# ── CELL 9: Results Comparison Table ─────────────────────────
metrics_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC']
keys = ['acc', 'f1', 'prec', 'rec', 'auc']

results_df = pd.DataFrame({
    'Metric': metrics_labels,
    'Baseline LR': [f"{np.mean(base_metrics[k]):.4f}" for k in keys],
    'PSO + LR':    [f"{np.mean(pso_metrics[k]):.4f}" for k in keys],
})

print("\n📋 Final Comparison:")
print(results_df.to_string(index=False))

print(f"\n📌 Features used — Baseline LR: {N_FEATURES} | PSO+LR: {len(selected_features)}")
print(f"📌 Feature reduction: {((N_FEATURES - len(selected_features)) / N_FEATURES * 100):.1f}%")


# ── CELL 10: Plots ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: metric comparison
x = np.arange(len(metrics_labels))
width = 0.35
base_vals = [np.mean(base_metrics[k]) for k in keys]
pso_vals  = [np.mean(pso_metrics[k]) for k in keys]

axes[0].bar(x - width/2, base_vals, width, label='Baseline LR', color='steelblue')
axes[0].bar(x + width/2, pso_vals,  width, label='PSO + LR',    color='darkorange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_labels, rotation=15)
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Baseline LR vs PSO+LR — Performance Comparison')
axes[0].legend()
axes[0].set_ylabel('Score')

# PSO convergence curve
axes[1].plot(optimizer.cost_history, color='darkorange', linewidth=2)
axes[1].set_title('PSO Convergence Curve')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Best Cost (1 - F1)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pso_lr_results.png', dpi=150)
plt.show()
print("✅ Plot saved as pso_lr_results.png")


# ── CELL 11: Confusion Matrices ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Retrain on 80% / test on 20% for confusion matrix display
from sklearn.model_selection import train_test_split

X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_scaled, y, test_size=0.2,
                                                     stratify=y, random_state=42)
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_pso,    y, test_size=0.2,
                                                     stratify=y, random_state=42)

pipeline_base.fit(X_tr_b, y_tr_b)
pipeline_pso.fit(X_tr_p, y_tr_p)

for ax, pipe, X_te, y_te, title in [
    (axes[0], pipeline_base, X_te_b, y_te_b, 'Baseline LR'),
    (axes[1], pipeline_pso,  X_te_p, y_te_p, 'PSO + LR'),
]:
    cm = confusion_matrix(y_te, pipe.predict(X_te))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("✅ Confusion matrices saved.")


# ── CELL 12: Save Artifacts ───────────────────────────────────
os.makedirs('artifacts', exist_ok=True)

joblib.dump(pipeline_base,    'artifacts/baseline_lr_pipeline.pkl')
joblib.dump(pipeline_pso,     'artifacts/pso_lr_pipeline.pkl')
joblib.dump(scaler,           'artifacts/scaler.pkl')
joblib.dump(selected_features,'artifacts/selected_features.pkl')
joblib.dump(feature_names,    'artifacts/feature_names.pkl')
joblib.dump(selected_names,   'artifacts/selected_names.pkl')
joblib.dump(base_metrics,     'artifacts/base_metrics.pkl')
joblib.dump(pso_metrics,      'artifacts/pso_metrics.pkl')

print("✅ All artifacts saved to /artifacts/")
print("Download them from the Colab file browser → right-click → Download")
