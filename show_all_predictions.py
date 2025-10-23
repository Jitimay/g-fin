import pandas as pd
import pickle
import numpy as np

def show_all_predictions():
    """Affiche les prédictions G-FIN pour tous les pays"""
    
    # Load model and data
    with open('model/gfin_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    
    df = pd.read_csv('data/gfin_real_data.csv')
    
    print("🛡️  G-FIN - PRÉDICTIONS GLOBALES 2022")
    print("=" * 60)
    
    # Get 2022 data for all countries
    latest_data = df[df['year'] == 2022].copy()
    
    predictions = []
    for _, row in latest_data.iterrows():
        X = row[features].values.reshape(1, -1)
        prob = model.predict_proba(X)[0, 1]
        immunity_score = 100 * (1 - prob)
        
        if immunity_score >= 80:
            risk_level = "🟢 STABLE"
        elif immunity_score >= 50:
            risk_level = "🟡 FRAGILE"
        else:
            risk_level = "🔴 HIGH RISK"
            
        predictions.append({
            'Country': row['country'],
            'Immunity_Score': immunity_score,
            'Risk_Level': risk_level,
            'Debt_GDP': row['debt_to_gdp'],
            'Inflation': row['inflation'],
            'Interest_Rate': row['interest_rate']
        })
    
    # Sort by immunity score
    predictions.sort(key=lambda x: x['Immunity_Score'])
    
    print(f"{'Pays':<15} {'Score':<8} {'Niveau':<12} {'Dette/PIB':<10} {'Inflation':<10} {'Taux':<8}")
    print("-" * 70)
    
    for pred in predictions:
        print(f"{pred['Country']:<15} {pred['Immunity_Score']:<8.1f} {pred['Risk_Level']:<12} "
              f"{pred['Debt_GDP']:<10.1f} {pred['Inflation']:<10.1f} {pred['Interest_Rate']:<8.1f}")
    
    # Summary
    high_risk = [p for p in predictions if p['Immunity_Score'] < 50]
    fragile = [p for p in predictions if 50 <= p['Immunity_Score'] < 80]
    stable = [p for p in predictions if p['Immunity_Score'] >= 80]
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ GLOBAL:")
    print(f"🔴 HIGH RISK ({len(high_risk)} pays): {', '.join([p['Country'] for p in high_risk])}")
    print(f"🟡 FRAGILE ({len(fragile)} pays): {', '.join([p['Country'] for p in fragile])}")
    print(f"🟢 STABLE ({len(stable)} pays): {', '.join([p['Country'] for p in stable])}")

if __name__ == "__main__":
    show_all_predictions()
