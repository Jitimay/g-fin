import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

# Load model and data
@st.cache_data
def load_data():
    return pd.read_csv('data/gfin_real_data.csv')

@st.cache_resource
def load_model():
    with open('model/gfin_model.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_immunity_score(model, features, data):
    X = np.array([data[f] for f in features]).reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    return 100 * (1 - prob)

# Page config
st.set_page_config(page_title="G-FIN - Global Financial Immunity Network", page_icon="🛡️")

st.title("🛡️ G-FIN - Global Financial Immunity Network")
st.markdown("AI-powered financial crisis prediction system")

# Load data and model
df = load_data()
model = load_model()
features = ['debt_to_gdp', 'fx_reserves', 'inflation', 'interest_rate', 
           'gdp_growth', 'export_revenue', 'budget_balance', 
           'political_stability', 'bond_yield_spread']

# Country selector
countries = df['country'].unique()
selected_country = st.selectbox("Select Country", countries)

# Get latest data for selected country
country_data = df[df['country'] == selected_country].iloc[-1]

# Calculate immunity score
score = calculate_immunity_score(model, features, country_data)

# Display results
col1, col2 = st.columns(2)
with col1:
    st.metric("Financial Immunity Score", f"{score:.1f}")
    
with col2:
    if score >= 80:
        st.success("🟢 STABLE - Low Risk")
    elif score >= 50:
        st.warning("🟡 FRAGILE - Medium Risk")
    else:
        st.error("🔴 HIGH RISK - Crisis Likely")

# Feature importance chart
feature_values = [country_data[f] for f in features]
fig = px.bar(x=features, y=feature_values, title="Financial Indicators")
st.plotly_chart(fig)

# Historical data
country_history = df[df['country'] == selected_country]
fig_history = px.line(country_history, x='year', y='debt_to_gdp', title="Debt-to-GDP Trend")
st.plotly_chart(fig_history)
