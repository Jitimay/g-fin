import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page config
st.set_page_config(
    page_title="DCEWS - Debt Crisis Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 10px; border-radius: 5px;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-warning {color: #ffa500; font-weight: bold;}
    .risk-stable {color: #00ff00; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('data/debt_data_100k.csv')

def load_model():
    with open('model/debt_model.pkl', 'rb') as f:
        return pickle.load(f)

def get_risk_level(prob):
    if prob >= 0.7:
        return "🔴 High Risk", "risk-high"
    elif prob >= 0.4:
        return "🟡 Warning", "risk-warning"
    else:
        return "🟢 Stable", "risk-stable"

def predict_country_risk(df, model_data):
    model = model_data['model']
    features = model_data['features']
    
    latest_data = df.groupby('country').last().reset_index()
    X = latest_data[features]
    probs = model.predict_proba(X)[:, 1]
    
    latest_data['default_prob'] = probs
    latest_data['risk_level'] = latest_data['default_prob'].apply(lambda x: get_risk_level(x)[0])
    
    return latest_data

def main():
    st.title("🚨 Debt Crisis Early Warning System")
    st.markdown("**AI-powered sovereign debt default prediction for African countries**")
    
    # Load data and model
    df = load_data()
    
    if not os.path.exists('model/debt_model.pkl'):
        st.error("Model not found. Please run: `python train_model.py`")
        return
    
    model_data = load_model()
    predictions = predict_country_risk(df, model_data)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    selected_country = st.sidebar.selectbox("Select Country", predictions['country'].unique())
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    high_risk = len(predictions[predictions['default_prob'] >= 0.7])
    warning = len(predictions[(predictions['default_prob'] >= 0.4) & (predictions['default_prob'] < 0.7)])
    stable = len(predictions[predictions['default_prob'] < 0.4])
    
    with col1:
        st.metric("🔴 High Risk", high_risk)
    with col2:
        st.metric("🟡 Warning", warning)
    with col3:
        st.metric("🟢 Stable", stable)
    
    # World map
    st.subheader("🗺️ Risk Map")
    
    # Create choropleth map
    fig = px.choropleth(
        predictions,
        locations='country',
        locationmode='country names',
        color='default_prob',
        hover_name='country',
        hover_data={'default_prob': ':.2f'},
        color_continuous_scale=['green', 'yellow', 'red'],
        range_color=[0, 1],
        title="Default Risk by Country"
    )
    fig.update_layout(
        geo=dict(scope='africa'),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Country details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"📊 {selected_country} Analysis")
        
        country_data = predictions[predictions['country'] == selected_country].iloc[0]
        prob = country_data['default_prob']
        risk_text, risk_class = get_risk_level(prob)
        
        st.markdown(f"**Default Probability:** {prob:.2%}")
        st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_text}</span>", unsafe_allow_html=True)
        
        # Key metrics
        st.write("**Key Indicators:**")
        st.write(f"• Debt-to-GDP: {country_data['debt_to_gdp']:.1f}%")
        st.write(f"• FX Reserves: ${country_data['fx_reserves']:.1f}B")
        st.write(f"• Inflation: {country_data['inflation']:.1f}%")
        st.write(f"• Bond Spread: {country_data['bond_yield_spread']:.1f}%")
        
        # Top risk factors
        st.write("**🔍 Top Risk Factors:**")
        feature_importance = model_data['model'].feature_importances_
        features = model_data['features']
        
        top_features = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)[:3]
        for i, (feature, importance) in enumerate(top_features, 1):
            st.write(f"{i}. {feature.replace('_', ' ').title()}: {importance:.2%} impact")
    
    with col2:
        st.subheader("📈 Historical Trends")
        
        country_history = df[df['country'] == selected_country]
        
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=country_history['year'],
            y=country_history['debt_to_gdp'],
            mode='lines+markers',
            name='Debt-to-GDP %',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f"{selected_country} Debt Trend",
            xaxis_title="Year",
            yaxis_title="Debt-to-GDP (%)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk table
    st.subheader("📋 All Countries Risk Assessment")
    
    display_df = predictions[['country', 'default_prob', 'risk_level', 'debt_to_gdp', 'fx_reserves']].copy()
    display_df['default_prob'] = display_df['default_prob'].apply(lambda x: f"{x:.2%}")
    display_df.columns = ['Country', 'Default Probability', 'Risk Level', 'Debt-to-GDP %', 'FX Reserves ($B)']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Feature importance chart
    st.subheader("🎯 Model Feature Importance")
    
    importance_df = pd.DataFrame({
        'Feature': model_data['features'],
        'Importance': model_data['model'].feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance in Default Prediction"
    )
    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
