import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_data():
    """Load G-FIN real financial data"""
    return pd.read_csv('data/gfin_real_data.csv')

def preprocess(df):
    """Preprocess data for G-FIN model"""
    features = ['debt_to_gdp', 'fx_reserves', 'inflation', 'interest_rate', 
                'gdp_growth', 'export_revenue', 'budget_balance', 
                'political_stability', 'bond_yield_spread']
    X = df[features]
    y = df['default']
    return X, y, features

def calculate_immunity_score(probabilities):
    """Calculate Financial Immunity Score (0-100)"""
    return 100 * (1 - probabilities)

def train_gfin_model():
    """Train G-FIN model with proper regularization"""
    df = load_data()
    X, y, features = preprocess(df)
    
    # Add noise to prevent overfitting on clean data
    np.random.seed(42)
    X_noisy = X.copy()
    for col in X.columns:
        noise = np.random.normal(0, X[col].std() * 0.05, len(X))
        X_noisy[col] = X[col] + noise
    
    # Temporal split: train on 2010-2018, test on 2019-2022
    train_mask = df['year'] <= 2018
    test_mask = df['year'] > 2018
    
    X_train, X_test = X_noisy[train_mask], X_noisy[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GradientBoosting with regularization
    model = GradientBoostingClassifier(
        n_estimators=50,        # Reduced to prevent overfitting
        learning_rate=0.05,     # Lower learning rate
        max_depth=3,            # Shallower trees
        min_samples_split=10,   # More samples required to split
        min_samples_leaf=5,     # More samples in leaf nodes
        subsample=0.8,          # Use subset of data
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions and Financial Immunity Scores
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Smooth probabilities to avoid extremes
    y_proba_smooth = np.clip(y_proba, 0.05, 0.95)
    immunity_scores = calculate_immunity_score(y_proba_smooth)
    
    print("G-FIN Model Performance:")
    print(classification_report(y_test, y_pred))
    if len(np.unique(y_test)) > 1:
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    
    # Sample immunity scores
    test_df = df[test_mask].reset_index(drop=True)
    print("\nSample Financial Immunity Scores:")
    for i in range(min(10, len(immunity_scores))):
        country = test_df.loc[i, 'country']
        score = immunity_scores[i]
        risk_level = "STABLE" if score >= 80 else "FRAGILE" if score >= 50 else "HIGH RISK"
        print(f"{country}: {score:.1f} ({risk_level})")
    
    # Save model with scaler
    os.makedirs('model', exist_ok=True)
    with open('model/gfin_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'scaler': scaler,
            'features': features,
            'immunity_threshold': {'stable': 80, 'fragile': 50}
        }, f)
    
    print("\nG-FIN model saved to model/gfin_model.pkl")
    return model, features

if __name__ == "__main__":
    train_gfin_model()
