import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef)

# ---- 1. Download & combine (pandas reads directly from a URL) ----
red   = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
# If BITS Lab blocks that domain, use the Kaggle mirror instead:
# kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 (download + upload manually)

red['type'], white['type'] = 'red', 'white'
df = pd.concat([red, white], ignore_index=True)
df.columns = [c.replace(' ', '_') for c in df.columns]   # UCI headers have spaces — clean them

print(df.shape)                          # (6497, 13)
print(df['quality'].value_counts().sort_index())

# ---- 2. Binarize target ----
# Raw quality is 3-9; the extremes (3, 9) have too few rows for a reliable
# stratified split or meaningful per-class metrics. Threshold at the
# community-standard cutoff instead: 7+ = "good".
df['target'] = (df['quality'] >= 7).astype(int)
TARGET = 'target'

X = df.drop(columns=['quality', 'target'])
y = df[TARGET]
X = pd.get_dummies(X, drop_first=True)    # 'type' -> one 'type_white' column = 12 features total

# ---- 3. Split — pick YOUR OWN random_state, not 42 ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y)   # change 17 to your own number

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

test_export = X_test.copy()
test_export[TARGET] = y_test
test_export.to_csv('test_data.csv', index=False)

# ---- 4. All 5 models — tune n_neighbors / n_estimators to your own choice ----
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree'      : DecisionTreeClassifier(random_state=17),
    'kNN'                : KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes'        : GaussianNB(),
    'Random Forest'      : RandomForestClassifier(random_state=17, n_estimators=250),
}

def evaluate(y_true, y_pred, y_proba):
    return {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall'   : recall_score(y_true, y_pred, zero_division=0),
        'F1'       : f1_score(y_true, y_pred, zero_division=0),
        'MCC'      : matthews_corrcoef(y_true, y_pred),
        'AUC'      : roc_auc_score(y_true, y_proba[:, 1]),
    }

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    pred, proba = model.predict(X_test_sc), model.predict_proba(X_test_sc)
    results[name] = evaluate(y_test, pred, proba)
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')
    print(f"{name:20s} -> {results[name]}")

joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_columns.pkl')
pd.DataFrame(results).T.round(4).to_csv('model/comparison_results.csv')
