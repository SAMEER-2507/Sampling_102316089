!pip install -q -U imbalanced-learn scikit-learn

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

# --- load dataset (adjust path if needed) ---
path = 'Creditcard_data.csv'
df = pd.read_csv(path)
print("Loaded:", path, " shape:", df.shape)
target_col = "Class"

# --- split features / target ---
X = df.drop(columns=[target_col])
y = df[target_col]

# --- simple preprocessing ---
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

print("Features shape after preprocessing:", X.shape)
print("Original class distribution:", Counter(y))

# --- choose oversampler ---
sampler = SMOTE(random_state=42) 

# --- apply oversampling ---
X_res, y_res = sampler.fit_resample(X, y)
print("Resampled features shape:", X_res.shape)
print("Resampled class distribution:", Counter(y_res))

# --- save balanced dataset ---
df_balanced = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
out_path = 'Creditcard_data_balanced.csv'
df_balanced.to_csv(out_path, index=False)
print("Balanced dataset saved to:", out_path)

!pip install -q -U scikit-learn

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------- parameters (change if needed) --------
path = "Creditcard_data_balanced.csv"
sample_frac = 0.30        # fraction for each sample
random_state = 42
max_clusters = 10         # maximum clusters for cluster sampling (KMeans)
# ----------------------------------------------

np.random.seed(random_state)

# --- load ---
df = pd.read_csv(path)
n = len(df)
sample_size = int(np.ceil(n * sample_frac))

# Function: print summary
def summary(name, sample_df):
    print(f"\n{name} | rows: {len(sample_df)}")
    try:
        print("Class distribution:", dict(pd.Series(sample_df[target_col]).value_counts(normalize=True)))
    except Exception:
        print("Class distribution: target column not present or not suitable")

# 1) Simple Random Sampling
simple_random = df.sample(n=sample_size, replace=False, random_state=random_state).reset_index(drop=True)
simple_random.to_csv("sample_simple_random.csv", index=False)
summary("Simple Random Sampling", simple_random)

# 2) Systematic Sampling
# take sample_size items by selecting indices at approximately equal intervals
if sample_size >= n:
    systematic = df.copy().reset_index(drop=True)
else:
    step = n / sample_size
    start = np.random.uniform(0, step)
    indices = (start + np.arange(sample_size) * step).astype(int)
    # ensure indices are within bounds
    indices = np.mod(indices, n)
    systematic = df.iloc[indices].reset_index(drop=True)
systematic.to_csv("sample_systematic.csv", index=False)
summary("Systematic Sampling", systematic)

# 3) Stratified Sampling (by target column)
# If target column has too many unique values or is continuous, fallback to simple random
stratified_success = False
if df[target_col].nunique() < n and df[target_col].nunique() <= (sample_size):  # basic sanity
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_frac, random_state=random_state)
        for _, test_idx in sss.split(df, df[target_col]):
            stratified = df.iloc[test_idx].reset_index(drop=True)
        stratified.to_csv("sample_stratified.csv", index=False)
        stratified_success = True
        summary("Stratified Sampling", stratified)
    except Exception as e:
        stratified_success = False
        print("Stratified sampling failed:", e)

if not stratified_success:
    # fallback to proportion-preserving manual sample via groupby.sample
    try:
        stratified = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=random_state)
        ).reset_index(drop=True)
        # If rounding caused size mismatch, adjust by sampling additional rows/trim
        if len(stratified) > sample_size:
            stratified = stratified.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        elif len(stratified) < sample_size:
            # add random rows to meet exact size
            deficit = sample_size - len(stratified)
            extra = df.drop(stratified.index, errors='ignore').sample(n=deficit, random_state=random_state)
            stratified = pd.concat([stratified, extra]).reset_index(drop=True)
        stratified.to_csv("sample_stratified.csv", index=False)
        summary("Stratified Sampling (fallback)", stratified)
    except Exception as e:
        print("Stratified fallback failed:", e)

# 4) Cluster Sampling
# Create clusters using KMeans on numeric features and select whole clusters until sample_size reached.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols = [c for c in num_cols if c != target_col]
if len(num_cols) == 0:
    # if no numeric features, fall back to simple random for cluster sampling
    cluster_sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    cluster_sample.to_csv("sample_cluster.csv", index=False)
    summary("Cluster Sampling (fallback to random, no numeric features)", cluster_sample)
else:
    X_num = df[num_cols].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_num)
    k = min(max_clusters, max(2, int(np.sqrt(n))))  # reasonable cluster count
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(Xs)
    df_clustered = df.copy()
    df_clustered['_cluster'] = labels
    # randomly pick clusters until we reach or exceed desired size
    clusters = list(df_clustered['_cluster'].unique())
    np.random.shuffle(clusters)
    chosen = []
    cum = 0
    for c in clusters:
        chosen.append(c)
        cum += (df_clustered['_cluster'] == c).sum()
        if cum >= sample_size:
            break
    cluster_sample = df_clustered[df_clustered['_cluster'].isin(chosen)].drop(columns=['_cluster']).reset_index(drop=True)
    # If cluster selection overshot size, trim randomly (this preserves cluster membership mostly but ensures exact size)
    if len(cluster_sample) > sample_size:
        cluster_sample = cluster_sample.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    cluster_sample.to_csv("sample_cluster.csv", index=False)
    summary("Cluster Sampling (one-stage, KMeans clusters)", cluster_sample)

# 5) Bootstrap Sampling (sampling with replacement)
bootstrap = df.sample(n=sample_size, replace=True, random_state=random_state).reset_index(drop=True)
bootstrap.to_csv("sample_bootstrap.csv", index=False)
summary("Bootstrap Sampling", bootstrap)

print("\nSaved files:\n - sample_simple_random.csv\n - sample_systematic.csv\n - sample_stratified.csv\n - sample_cluster.csv\n - sample_bootstrap.csv")

!pip install -q -U scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ---------- USER-CONFIG ----------
sample_paths = [
    "sample_simple_random.csv",
    "sample_systematic.csv",
    "sample_stratified.csv",
    "sample_cluster.csv",
    "sample_bootstrap.csv"
]
test_size = 0.20
random_state = 42
# ---------------------------------

for sample_path in sample_paths:

    # Load
    df = pd.read_csv(sample_path)
    print("Loaded:", sample_path, "shape:", df.shape)

    # Heuristic detect target column (same as previous cells)
    candidates = ['Class','class','is_fraud','fraud','target','Target','label','Label','y','Y']
    target_col = None
    for c in candidates:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        binary_cols = [c for c in df.columns if df[c].nunique() == 2]
        if len(binary_cols) == 1:
            target_col = binary_cols[0]
        elif len(binary_cols) > 1:
            ratios = {c: df[c].value_counts(normalize=True).min() for c in binary_cols}
            target_col = min(ratios, key=ratios.get)
        else:
            target_col = df.columns[-1]
    print("Using target column:", target_col)

    # Prepare X, y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if needed
    if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print("Target encoded. Classes:", list(le.classes_))

    # Basic preprocessing: get dummies for object cols, scale numeric features
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Scale numeric columns
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    print("Features shape after preprocessing:", X.shape)
    print("Class distribution:", dict(pd.Series(y).value_counts(normalize=True)))

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )

    # Define models (pipelines where scaling might matter for SVC/KNN/Logistic)
    models = {
        "LogisticRegression": Pipeline([("clf", LogisticRegression(max_iter=2000, random_state=random_state))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        "SVM": Pipeline([("clf", SVC(probability=False, random_state=random_state))]),
        "KNeighbors": Pipeline([("clf", KNeighborsClassifier(n_neighbors=5))])
    }

    # Train and evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{name} accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, zero_division=0))
        results.append({"model": name, "accuracy": acc})

    # Summary dataframe and save
    res_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    out_name = f"models_accuracy_summary_{sample_path.replace('.csv','')}.csv"
    res_df.to_csv(out_name, index=False)
    print(f"\nSaved summary to {out_name}")
    print(res_df)
