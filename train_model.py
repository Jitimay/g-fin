import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

def load_data():
    return pd.read_csv('data/debt_data_100k.csv')

def preprocess(df):
    features = ['debt_to_gdp', 'fx_reserves', 'inflation', 'interest_rate', 
                'gdp_growth', 'export_revenue', 'budget_balance', 
                'political_stability', 'bond_yield_spread']
    X = df[features]
    y = df['default']
    return X, y, features

def train_model():
    df = load_data()
    X, y, features = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    
    # Save model
    os.makedirs('model', exist_ok=True)
    with open('model/debt_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)
    
    print("\nModel saved to model/debt_model.pkl")
    return model, features

if __name__ == "__main__":
    train_model()
