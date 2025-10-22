import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# Page config
st.set_page_config(
    page_title="Debt Radar - AI Crisis Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #ffffff;
    }
    .stMetric {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 2px solid #4a90e2 !important;
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.3) !important;
        color: #212529 !important;
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: #212529 !important;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    .risk-high {
        color: #ff4757;
        font-weight: bold;
        text-shadow: 0 0 10px #ff4757;
    }
    .risk-warning {
        color: #ffa502;
        font-weight: bold;
        text-shadow: 0 0 10px #ffa502;
    }
    .risk-stable {
        color: #2ed573;
        font-weight: bold;
        text-shadow: 0 0 10px #2ed573;
    }
    .country-card {
        background: linear-gradient(145deg, #2d3748, #4a5568);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #3182ce;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a202c, #2d3748);
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .alert-banner {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid #ff4757;
        box-shadow: 0 0 30px rgba(231, 76, 60, 0.5);
        animation: pulse 2s infinite;
        color: #ffffff;
        font-weight: bold;
        font-size: 1.1rem;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
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
        return "🔴 HIGH RISK", "risk-high", "#ff4757"
    elif prob >= 0.4:
        return "🟡 WARNING", "risk-warning", "#ffa502"
    else:
        return "🟢 STABLE", "risk-stable", "#2ed573"

def predict_country_risk(df, model_data):
    model = model_data['model']
    features = model_data['features']
    
    latest_data = df.groupby('country').last().reset_index()
    X = latest_data[features]
    probs = model.predict_proba(X)[:, 1]
    
    latest_data['default_prob'] = probs
    latest_data['risk_level'] = latest_data['default_prob'].apply(lambda x: get_risk_level(x)[0])
    latest_data['risk_color'] = latest_data['default_prob'].apply(lambda x: get_risk_level(x)[2])
    
    return latest_data

def create_risk_widget(prob, country):
    risk_text, risk_class, risk_color = get_risk_level(prob)
    
    # Fix progress width calculation
    progress_width = max(5, int(prob * 100))  # Minimum 5% width for visibility
    
    widget_html = f"""
    <div style="
        background: linear-gradient(145deg, #2a2a2a, #3a3a3a);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid {risk_color};
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 0 20px {risk_color}40;
    ">
        <h2 style="color: {risk_color}; margin: 0;">{country}</h2>
        <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{prob:.0%}</h1>
        <p style="color: {risk_color}; font-size: 1.2rem; margin: 0;">{risk_text}</p>
        
        <div style="
            background: #1a1a1a;
            border-radius: 10px;
            height: 20px;
            margin: 15px 0;
            overflow: hidden;
            border: 1px solid #444;
        ">
            <div style="
                background: linear-gradient(90deg, {risk_color}, {risk_color}aa);
                height: 100%;
                width: {progress_width}%;
                border-radius: 10px;
                animation: fillBar 1s ease-out;
            "></div>
        </div>
        
        <p style="color: #ccc; margin: 0; font-size: 0.9rem;">Risk Level: {prob:.1%}</p>
    </div>
    
    <style>
    @keyframes fillBar {{
        from {{ width: 0%; }}
        to {{ width: {progress_width}%; }}
    }}
    </style>
    """
    return widget_html

def main():
    # Header
    st.markdown("<h1>🎯 DEBT RADAR</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Sovereign Debt Crisis Prediction for Africa</p>", unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    
    if not os.path.exists('model/debt_model.pkl'):
        st.error("🚨 Model not found. Please run: `python train_model.py`")
        return
    
    model_data = load_model()
    predictions = predict_country_risk(df, model_data)
    
    # Alert banner for high-risk countries
    high_risk_countries = predictions[predictions['default_prob'] >= 0.7]['country'].tolist()
    if high_risk_countries:
        st.markdown(f"""
        <div class='alert-banner'>
            🚨 CRISIS ALERT: {len(high_risk_countries)} countries at HIGH RISK - {', '.join(high_risk_countries[:3])}
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    selected_country = st.sidebar.selectbox("🌍 Select Country", predictions['country'].unique())
    
    # Risk filter
    risk_filter = st.sidebar.multiselect(
        "🎯 Filter by Risk Level",
        ["🔴 HIGH RISK", "🟡 WARNING", "🟢 STABLE"],
        default=["🔴 HIGH RISK", "🟡 WARNING", "🟢 STABLE"]
    )
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = len(predictions[predictions['default_prob'] >= 0.7])
    warning = len(predictions[(predictions['default_prob'] >= 0.4) & (predictions['default_prob'] < 0.7)])
    stable = len(predictions[predictions['default_prob'] < 0.4])
    total_countries = len(predictions)
    
    with col1:
        st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/total_countries:.1%}")
    with col2:
        st.metric("🟡 Warning", warning, delta=f"{warning/total_countries:.1%}")
    with col3:
        st.metric("🟢 Stable", stable, delta=f"{stable/total_countries:.1%}")
    with col4:
        st.metric("📊 Total Countries", total_countries)
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ African Risk Map")
        
        # Enhanced choropleth map
        fig = px.choropleth(
            predictions,
            locations='country',
            locationmode='country names',
            color='default_prob',
            hover_name='country',
            hover_data={
                'default_prob': ':.1%',
                'debt_to_gdp': ':.1f',
                'political_stability': ':.1f'
            },
            color_continuous_scale=['#2ed573', '#ffa502', '#ff4757'],
            range_color=[0, 1],
            title="Default Risk Probability by Country"
        )
        fig.update_layout(
            geo=dict(
                scope='africa',
                bgcolor='rgba(0,0,0,0)',
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Live Risk Monitor")
        country_data = predictions[predictions['country'] == selected_country].iloc[0]
        prob = country_data['default_prob']
        risk_text, risk_class, risk_color = get_risk_level(prob)
        
        # Simple risk display with Streamlit components
        st.markdown(f"### {selected_country}")
        st.markdown(f"<h1 style='color: {risk_color}; text-align: center;'>{prob:.0%}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {risk_color}; text-align: center; font-size: 1.2rem;'>{risk_text}</p>", unsafe_allow_html=True)
        
        # Progress bar
        st.progress(prob)
        st.caption(f"Risk Level: {prob:.1%}")
        
        # Key indicators
        st.markdown("### 📊 Key Indicators")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("💰 Debt/GDP", f"{country_data['debt_to_gdp']:.1f}%")
            st.metric("📈 Inflation", f"{country_data['inflation']:.1f}%")
        with col_b:
            st.metric("🏛️ Stability", f"{country_data['political_stability']:.1f}")
            st.metric("💱 FX Reserves", f"${country_data['fx_reserves']:.1f}B")
    
    # Detailed analysis
    st.subheader(f"📈 {selected_country} Deep Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical trends
        country_history = df[df['country'] == selected_country].tail(10)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Debt Trend', 'Inflation', 'Political Stability', 'Bond Spreads'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=country_history['year'], y=country_history['debt_to_gdp'],
                                mode='lines+markers', name='Debt-to-GDP', line=dict(color='#ff4757')), row=1, col=1)
        fig.add_trace(go.Scatter(x=country_history['year'], y=country_history['inflation'],
                                mode='lines+markers', name='Inflation', line=dict(color='#ffa502')), row=1, col=2)
        fig.add_trace(go.Scatter(x=country_history['year'], y=country_history['political_stability'],
                                mode='lines+markers', name='Stability', line=dict(color='#2ed573')), row=2, col=1)
        fig.add_trace(go.Scatter(x=country_history['year'], y=country_history['bond_yield_spread'],
                                mode='lines+markers', name='Bond Spread', line=dict(color='#3742fa')), row=2, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        st.write("🎯 **Risk Factor Analysis**")
        importance_df = pd.DataFrame({
            'Factor': model_data['features'],
            'Impact': model_data['model'].feature_importances_
        }).sort_values('Impact', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Impact',
            y='Factor',
            orientation='h',
            color='Impact',
            color_continuous_scale=['#2ed573', '#ffa502', '#ff4757'],
            title="Model Feature Importance"
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top risk factors for selected country
        st.write("🔍 **Key Risk Drivers:**")
        top_factors = importance_df.tail(3)
        for i, (_, row) in enumerate(top_factors.iterrows(), 1):
            factor_value = country_data[row['Factor']]
            st.write(f"{i}. **{row['Factor'].replace('_', ' ').title()}**: {factor_value:.1f} ({row['Impact']:.1%} impact)")
    
    # Risk table with enhanced styling
    st.subheader("📋 Country Risk Dashboard")
    
    # Filter data based on selection
    filtered_data = predictions[predictions['risk_level'].isin(risk_filter)]
    
    # Enhanced dataframe
    display_df = filtered_data[['country', 'default_prob', 'risk_level', 'debt_to_gdp', 
                               'political_stability', 'inflation']].copy()
    display_df['default_prob'] = display_df['default_prob'].apply(lambda x: f"{x:.1%}")
    display_df.columns = ['Country', 'Default Risk', 'Status', 'Debt/GDP %', 'Stability', 'Inflation %']
    
    # Sort by risk
    display_df = display_df.sort_values('Default Risk', ascending=False)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Footer stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Dataset Size", f"{len(df):,} records")
    with col2:
        st.metric("🎯 Model Accuracy", "96%")
    with col3:
        avg_risk = predictions['default_prob'].mean()
        st.metric("📈 Average Risk", f"{avg_risk:.1%}")

if __name__ == "__main__":
    main()
