#!/usr/bin/env python3
"""
G-FIN Demo - Global Financial Immunity Network
Test predictions for Sri Lanka, Ghana, and Zambia
"""

import pandas as pd
import pickle
import numpy as np

def load_gfin_model():
    """Load G-FIN model"""
    with open('model/gfin_model.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_immunity_score(model, features, data):
    """Calculate Financial Immunity Score"""
    X = np.array([data[f] for f in features]).reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    return 100 * (1 - prob)

def get_risk_assessment(score):
    """Get risk assessment from immunity score"""
    if score >= 80:
        return "🟢 STABLE", "Low risk - Strong financial immunity"
    elif score >= 50:
        return "🟡 FRAGILE", "Medium risk - Vulnerable to shocks"
    else:
        return "🔴 HIGH RISK", "High risk - Crisis likely within 6-12 months"

def demo_gfin():
    """Run G-FIN demo with real country data"""
    print("🛡️  G-FIN - Global Financial Immunity Network")
    print("=" * 50)
    print("AI-Powered Financial Crisis Prediction System")
    print("Predicting crises 6-12 months in advance\n")
    
    # Load model
    model_data = load_gfin_model()
    model = model_data['model']
    features = model_data['features']
    
    # Test cases based on real data
    test_cases = {
        "Sri Lanka 2022 (Crisis Year)": {
            'debt_to_gdp': 128.0,
            'fx_reserves': 1.9,
            'inflation': 46.0,
            'interest_rate': 28.0,
            'gdp_growth': -3.6,
            'export_revenue': 13.1,
            'budget_balance': -10.0,
            'political_stability': -1.5,
            'bond_yield_spread': 25.0
        },
        "Ghana 2022 (Crisis Year)": {
            'debt_to_gdp': 72.9,
            'fx_reserves': 6.3,
            'inflation': 54.1,
            'interest_rate': 27.0,
            'gdp_growth': 3.1,
            'export_revenue': 17.5,
            'budget_balance': -6.0,
            'political_stability': -0.1,
            'bond_yield_spread': 20.0
        },
        "Zambia 2020 (Pre-Default)": {
            'debt_to_gdp': 119.0,
            'fx_reserves': 1.4,
            'inflation': 15.7,
            'interest_rate': 20.0,
            'gdp_growth': -2.8,
            'export_revenue': 8.0,
            'budget_balance': -7.0,
            'political_stability': -0.2,
            'bond_yield_spread': 18.0
        },
        "Kenya 2022 (Stable)": {
            'debt_to_gdp': 65.0,
            'fx_reserves': 8.5,
            'inflation': 7.5,
            'interest_rate': 8.5,
            'gdp_growth': 5.2,
            'export_revenue': 12.0,
            'budget_balance': -4.5,
            'political_stability': 0.1,
            'bond_yield_spread': 6.5
        }
    }
    
    print("🔍 FINANCIAL IMMUNITY ANALYSIS")
    print("-" * 50)
    
    for country_case, data in test_cases.items():
        immunity_score = calculate_immunity_score(model, features, data)
        risk_level, description = get_risk_assessment(immunity_score)
        
        print(f"\n📊 {country_case}")
        print(f"   Financial Immunity Score: {immunity_score:.1f}/100")
        print(f"   Risk Level: {risk_level}")
        print(f"   Assessment: {description}")
        
        # Show key risk factors
        print("   Key Indicators:")
        print(f"   • Debt/GDP: {data['debt_to_gdp']:.1f}%")
        print(f"   • FX Reserves: ${data['fx_reserves']:.1f}B")
        print(f"   • Inflation: {data['inflation']:.1f}%")
        print(f"   • Bond Spread: {data['bond_yield_spread']:.1f}%")
    
    print("\n" + "=" * 50)
    print("🚨 G-FIN ALERT SYSTEM")
    print("-" * 50)
    
    # Simulate alerts
    for country_case, data in test_cases.items():
        immunity_score = calculate_immunity_score(model, features, data)
        if immunity_score < 50:
            country_name = country_case.split()[0]
            print(f"🚨 ALERT: {country_name} - Immunity Score: {immunity_score:.1f}")
            print(f"   Recommended Action: Immediate policy intervention required")
        elif immunity_score < 80:
            country_name = country_case.split()[0]
            print(f"⚠️  WARNING: {country_name} - Immunity Score: {immunity_score:.1f}")
            print(f"   Recommended Action: Monitor closely, prepare contingency plans")
    
    print("\n🎯 Model Feature Importance:")
    print("-" * 30)
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.3f}")
    
    print(f"\n✅ G-FIN Demo Complete")
    print(f"   Model trained on {len(features)} financial indicators")
    print(f"   Predicting crises 6-12 months in advance")
    print(f"   Ready for real-time monitoring and alerts")

if __name__ == "__main__":
    demo_gfin()
