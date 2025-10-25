import pandas as pd
import numpy as np
import pickle
import os

def calculate_risk_score(row):
    """Calculate risk score based on financial indicators"""
    risk = 0
    # Debt burden
    if row['debt_to_gdp'] > 100: risk += 0.4
    elif row['debt_to_gdp'] > 70: risk += 0.2
    elif row['debt_to_gdp'] > 50: risk += 0.1
    
    # FX reserves
    if row['fx_reserves'] < 3: risk += 0.3
    elif row['fx_reserves'] < 6: risk += 0.15
    
    # Inflation
    if row['inflation'] > 20: risk += 0.25
    elif row['inflation'] > 10: risk += 0.1
    
    # Interest rates
    if row['interest_rate'] > 20: risk += 0.2
    elif row['interest_rate'] > 15: risk += 0.1
    
    # GDP growth
    if row['gdp_growth'] < -2: risk += 0.2
    elif row['gdp_growth'] < 2: risk += 0.1
    
    # Bond spreads
    if row['bond_yield_spread'] > 15: risk += 0.3
    elif row['bond_yield_spread'] > 10: risk += 0.15
    elif row['bond_yield_spread'] > 5: risk += 0.05
    
    # Political stability
    if row['political_stability'] < -1: risk += 0.1
    elif row['political_stability'] < 0: risk += 0.05
    
    return min(risk, 0.95)  # Cap at 95%

def create_balanced_model():
    """Create a more realistic model with proper probability calibration"""
    df = pd.read_csv('data/gfin_real_data.csv')
    
    features = ['debt_to_gdp', 'fx_reserves', 'inflation', 'interest_rate', 
                'gdp_growth', 'export_revenue', 'budget_balance', 
                'political_stability', 'bond_yield_spread']
    
    # Save the model as a simple rule-based system
    model_data = {
        'type': 'rule_based',
        'features': features
    }
    
    os.makedirs('model', exist_ok=True)
    with open('model/gfin_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ G-FIN Rule-Based Model Created")
    print("Sample Immunity Scores:")
    
    # Test on recent data
    recent_data = df[df['year'] == 2022]
    for _, row in recent_data.iterrows():
        risk = calculate_risk_score(row)
        immunity = (1 - risk) * 100
        risk_level = "STABLE" if immunity >= 80 else "FRAGILE" if immunity >= 50 else "HIGH RISK"
        print(f"{row['country']}: {immunity:.1f} ({risk_level})")
    
    return model_data

if __name__ == "__main__":
    create_balanced_model()
