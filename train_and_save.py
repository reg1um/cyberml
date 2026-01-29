import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from lightgbm import LGBMClassifier

# Configuration
CACHE_FILE = "sampled_data.pkl"
RANDOM_STATE = 42

def load_data():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        return pd.read_pickle(CACHE_FILE)
    else:
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found.")

def preprocess(df):
    df['is_attack'] = (df['Label'] != 'Benign').astype(int)
    y_binary = df['is_attack'].values
    y_multi_raw = df['Label'].values
    
    df_features = df.drop(columns=['Label', 'is_attack'])
    
    non_numeric_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop = [col for col in non_numeric_cols if any(x in col.lower() for x in ['ip', 'address', 'id', 'time', 'stamp'])]
    df_features = df_features.drop(columns=cols_to_drop, errors='ignore')
    
    le = LabelEncoder()
    for col in df_features.select_dtypes(include=['object']).columns:
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        
    df_features = df_features.fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features.values)
    
    le_multi = LabelEncoder()
    y_multi = le_multi.fit_transform(y_multi_raw)
    
    return X, y_binary, y_multi, le_multi.classes_, scaler, le_multi

def run():
    print("Loading Data...")
    df = load_data()
    X, y_binary, y_multi, class_names, scaler, le_multi = preprocess(df)
    
    print(f"Classes: {list(class_names)}")
    
    print("Splitting data...")
    indices = np.arange(len(X))
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, idx_train, idx_test = train_test_split(
        X, y_binary, y_multi, indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multi
    )
    
    # Save test set for demo
    test_df = df.iloc[idx_test].copy()
    test_df.to_pickle('test_data.pkl')
    print(f"Saved test set ({len(test_df)} samples) to test_data.pkl")
    
    # Train LightGBM (Multi-class classification - same params as benchmark)
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=800, max_depth=15, learning_rate=0.03,
        num_leaves=100, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced', verbose=-1
    )
    lgbm.fit(X_train, y_train_multi)
    
    y_pred = lgbm.predict(X_test)
    mcc = matthews_corrcoef(y_test_multi, y_pred)
    print(f"LightGBM MCC: {mcc:.4f}")
    
    print("Saving artifacts...")
    artifacts = {
        'model': lgbm,
        'scaler': scaler,
        'label_encoder': le_multi,
        'class_names': class_names
    }
    joblib.dump(artifacts, 'model_artifacts.pkl')
    print("Done!")

if __name__ == "__main__":
    run()
