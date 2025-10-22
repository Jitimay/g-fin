#!/usr/bin/env python3
"""
DCEWS Demo Script - Shows the system predictions
"""
import pandas as pd
import pickle

def main():
    print("🚨 DEBT CRISIS EARLY WARNING SYSTEM (DCEWS)")
    print("=" * 50)
    
    # Load data and model
    df = pd.read_csv('data/debt_data.csv')
    with open('model/debt_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features = model_data['features']
    
    # Get latest data for each country
    latest_data = df.groupby('country').last().reset_index()
    X = latest_data[features]
    probs = model.predict_proba(X)[:, 1]
    
    # Add predictions
    latest_data['default_prob'] = probs
    
    def get_risk_level(prob):
        if prob >= 0.7:
            return "🔴 HIGH RISK"
        elif prob >= 0.4:
            return "🟡 WARNING"
        else:
            return "🟢 STABLE"
    
    latest_data['risk_level'] = latest_data['default_prob'].apply(get_risk_level)
    
    # Sort by risk
    latest_data = latest_data.sort_values('default_prob', ascending=False)
    
    print("\n📊 COUNTRY RISK RANKINGS (2023)")
    print("-" * 70)
    print(f"{'Country':<12} {'Risk Level':<12} {'Probability':<12} {'Debt/GDP':<10}")
    print("-" * 70)
    
    for _, row in latest_data.iterrows():
        prob_str = f"{row['default_prob']:.1%}"
        print(f"{row['country']:<12} {row['risk_level']:<12} {prob_str:<12} {row['debt_to_gdp']:.1f}%")
    
    print("\n🔍 TOP RISK FACTORS:")
    importance = pd.DataFrame({
        'Factor': features,
        'Impact': model.feature_importances_
    }).sort_values('Impact', ascending=False)
    
    for i, (_, row) in enumerate(importance.head(5).iterrows(), 1):
        print(f"{i}. {row['Factor'].replace('_', ' ').title()}: {row['Impact']:.1%}")
    
    print(f"\n🎯 SUMMARY:")
    high_risk = len(latest_data[latest_data['default_prob'] >= 0.7])
    warning = len(latest_data[(latest_data['default_prob'] >= 0.4) & (latest_data['default_prob'] < 0.7)])
    stable = len(latest_data[latest_data['default_prob'] < 0.4])
    
    print(f"🔴 High Risk Countries: {high_risk}")
    print(f"🟡 Warning Countries: {warning}")
    print(f"🟢 Stable Countries: {stable}")
    
    print(f"\n🚀 To launch interactive dashboard:")
    print(f"   streamlit run app.py")

if __name__ == "__main__":
    main()
