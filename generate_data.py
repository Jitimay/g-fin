import pandas as pd
import numpy as np

# Extended list of African countries including Burundi
countries = [
    'Kenya', 'Ghana', 'Zambia', 'Nigeria', 'South Africa', 'Tanzania', 
    'Rwanda', 'Ethiopia', 'Senegal', 'Uganda', 'Burundi', 'Mali', 
    'Burkina Faso', 'Chad', 'Cameroon', 'Madagascar', 'Mozambique',
    'Angola', 'Zimbabwe', 'Botswana', 'Namibia', 'Malawi', 'Benin',
    'Togo', 'Guinea', 'Sierra Leone', 'Liberia', 'Gabon', 'Congo',
    'Central African Republic'
]

years = list(range(2000, 2024))
n_records = 100000

np.random.seed(42)
data = []

for i in range(n_records):
    country = np.random.choice(countries)
    year = np.random.choice(years)
    
    # Base risk factors by country type
    if country in ['Burundi', 'Chad', 'Central African Republic', 'Mali']:
        # High-risk profile
        debt_base, inflation_base, stability_base = 85, 15, -0.8
    elif country in ['Ghana', 'Zambia', 'Ethiopia']:
        # Medium-high risk
        debt_base, inflation_base, stability_base = 75, 12, -0.4
    elif country in ['South Africa', 'Botswana', 'Namibia']:
        # Stable economies
        debt_base, inflation_base, stability_base = 45, 5, 0.3
    else:
        # Average risk
        debt_base, inflation_base, stability_base = 60, 8, 0.0
    
    # Add year trends and noise
    year_factor = (year - 2020) * 0.5
    
    debt_to_gdp = max(20, debt_base + year_factor + np.random.normal(0, 10))
    fx_reserves = max(0.5, np.random.exponential(5) + np.random.normal(0, 2))
    inflation = max(0, inflation_base + np.random.normal(0, 5))
    interest_rate = max(1, inflation + np.random.normal(2, 3))
    gdp_growth = np.random.normal(3, 4)
    export_revenue = max(1, np.random.exponential(10))
    budget_balance = np.random.normal(-5, 3)
    political_stability = max(-2, min(1, stability_base + np.random.normal(0, 0.3)))
    bond_yield_spread = max(0.5, debt_to_gdp/10 + inflation/5 + np.random.normal(0, 2))
    
    # Default probability based on risk factors
    risk_score = (debt_to_gdp/100 * 0.3 + 
                 max(0, inflation-5)/20 * 0.2 + 
                 max(0, -political_stability) * 0.3 + 
                 bond_yield_spread/20 * 0.2)
    
    default = 1 if risk_score > 0.6 and np.random.random() < risk_score else 0
    
    data.append([
        country, year, round(debt_to_gdp, 1), round(fx_reserves, 1),
        round(inflation, 1), round(interest_rate, 1), round(gdp_growth, 1),
        round(export_revenue, 1), round(budget_balance, 1), 
        round(political_stability, 1), round(bond_yield_spread, 1), default
    ])

df = pd.DataFrame(data, columns=[
    'country', 'year', 'debt_to_gdp', 'fx_reserves', 'inflation',
    'interest_rate', 'gdp_growth', 'export_revenue', 'budget_balance',
    'political_stability', 'bond_yield_spread', 'default'
])

df.to_csv('data/debt_data_100k.csv', index=False)
print(f"Generated {len(df)} records for {len(countries)} countries")
print(f"Burundi records: {len(df[df['country'] == 'Burundi'])}")
print(f"Default rate: {df['default'].mean():.1%}")
