import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
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
    """Train G-FIN GradientBoosting model with temporal validation"""
    df = load_data()
    X, y, features = preprocess(df)
    
    # Temporal split: train on 2010-2018, test on 2019-2022
    train_mask = df['year'] <= 2018
    test_mask = df['year'] > 2018
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # GradientBoosting model (similar to XGBoost)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions and Financial Immunity Scores
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    immunity_scores = calculate_immunity_score(y_proba)
    
    print("G-FIN Model Performance:")
    print(classification_report(y_test, y_pred))
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
    
    # Save model
    os.makedirs('model', exist_ok=True)
    with open('model/gfin_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'features': features,
            'immunity_threshold': {'stable': 80, 'fragile': 50}
        }, f)
    
    print("\nG-FIN model saved to model/gfin_model.pkl")
    return model, features

if __name__ == "__main__":
    train_gfin_model()
