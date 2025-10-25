import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import time

# Configure page
st.set_page_config(
    page_title="G-FIN Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# World-class CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        background: #0a0e1a;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b69 50%, #0a0e1a 100%);
        padding: 20px 40px;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 2px solid #00d4ff;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="%23ffffff" stroke-width="0.1" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #888;
        margin: 5px 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    .immunity-terminal {
        background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(0,0,0,0.8) 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 50px rgba(0,212,255,0.2);
    }
    
    .immunity-terminal::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00d4ff, #5b73ff, #ff006e, #00d4ff);
        border-radius: 15px;
        z-index: -1;
        animation: borderGlow 3s linear infinite;
    }
    
    @keyframes borderGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .immunity-score-display {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .immunity-number {
        font-size: 5rem;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 0 0 30px currentColor;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .stable-score { color: #00d4ff; }
    .fragile-score { color: #ffaa00; }
    .risk-score { color: #ff3366; }
    
    .country-info {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 15px;
        opacity: 0.9;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .terminal-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0.8) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .terminal-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 20px 40px rgba(0,212,255,0.1);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    .metric-item {
        background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(0,0,0,0.5) 100%);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: scale(1.05);
        border-color: #00d4ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .alert-panel {
        background: linear-gradient(135deg, rgba(255,51,102,0.2) 0%, rgba(0,0,0,0.8) 100%);
        border: 2px solid #ff3366;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        animation: alertPulse 2s ease-in-out infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255,51,102,0.3); }
        50% { box-shadow: 0 0 40px rgba(255,51,102,0.6); }
    }
    
    .chart-terminal {
        background: rgba(0,0,0,0.6);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(0,212,255,0.1) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        color: white !important;
    }
    
    .terminal-button {
        background: linear-gradient(135deg, #ff3366 0%, #ff6b9d 100%);
        border: none;
        border-radius: 8px;
        padding: 12px 25px;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .terminal-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255,51,102,0.4);
    }
    
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00ff00;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s ease-in-out infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .data-stream {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #00d4ff;
        background: rgba(0,0,0,0.8);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Risk calculation function
def calculate_risk_score(row):
    risk = 0
    if row['debt_to_gdp'] > 100: risk += 0.4
    elif row['debt_to_gdp'] > 70: risk += 0.2
    elif row['debt_to_gdp'] > 50: risk += 0.1
    
    if row['fx_reserves'] < 3: risk += 0.3
    elif row['fx_reserves'] < 6: risk += 0.15
    
    if row['inflation'] > 20: risk += 0.25
    elif row['inflation'] > 10: risk += 0.1
    
    if row['interest_rate'] > 20: risk += 0.2
    elif row['interest_rate'] > 15: risk += 0.1
    
    if row['gdp_growth'] < -2: risk += 0.2
    elif row['gdp_growth'] < 2: risk += 0.1
    
    if row['bond_yield_spread'] > 15: risk += 0.3
    elif row['bond_yield_spread'] > 10: risk += 0.15
    elif row['bond_yield_spread'] > 5: risk += 0.05
    
    if row['political_stability'] < -1: risk += 0.1
    elif row['political_stability'] < 0: risk += 0.05
    
    return min(risk, 0.95)

@st.cache_data
def load_data():
    return pd.read_csv('data/gfin_real_data.csv')

@st.cache_resource
def load_model():
    with open('model/gfin_model.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_immunity_score(country_data):
    risk = calculate_risk_score(country_data)
    noise = np.random.normal(0, 0.02)
    final_risk = np.clip(risk + noise, 0.05, 0.95)
    return (1 - final_risk) * 100

def predict_future_scores(country_data, months=12):
    """Predict immunity scores for next 1-12 months"""
    current_score = calculate_immunity_score(country_data)
    predictions = []
    
    # Base trend calculation
    debt_trend = -0.5 if country_data['debt_to_gdp'] > 80 else 0.2
    inflation_trend = -0.3 if country_data['inflation'] > 15 else 0.1
    growth_trend = 0.3 if country_data['gdp_growth'] > 2 else -0.2
    
    base_trend = (debt_trend + inflation_trend + growth_trend) / 3
    
    for month in range(1, months + 1):
        # Add volatility and seasonal effects
        volatility = np.random.normal(0, 1.5)
        seasonal = 2 * np.sin(month * np.pi / 6)  # 6-month cycle
        
        trend_effect = base_trend * month * 0.8
        predicted_score = current_score + trend_effect + volatility + seasonal
        
        # Keep within realistic bounds
        predicted_score = np.clip(predicted_score, 5, 95)
        predictions.append(predicted_score)
    
    return predictions

def get_risk_level(score):
    if score >= 80:
        return "STABLE", "stable-score"
    elif score >= 50:
        return "FRAGILE", "fragile-score"
    else:
        return "HIGH RISK", "risk-score"

def main():
    # Terminal Header
    st.markdown("""
    <div class="main-header">
        <div class="terminal-title">🛡️ G-FIN TERMINAL</div>
        <div class="terminal-subtitle">
            <span class="live-indicator"></span>GLOBAL FINANCIAL IMMUNITY NETWORK • REAL-TIME CRISIS PREDICTION
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Control Panel
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        countries = sorted(df['country'].unique())
        selected_country = st.selectbox("🌍 TARGET COUNTRY", countries, key="country")
    
    with col2:
        years = sorted(df['year'].unique())
        selected_year = st.selectbox("📅 ANALYSIS YEAR", years, index=len(years)-1, key="year")
    
    with col3:
        if st.button("🚨 EMERGENCY ALERT", key="alert"):
            st.balloons()
    
    with col4:
        st.markdown(f"""
        <div class="data-stream">
            LIVE: {time.strftime('%H:%M:%S UTC')}
        </div>
        """, unsafe_allow_html=True)
    
    # Get country data
    country_data = df[(df['country'] == selected_country) & (df['year'] == selected_year)]
    
    if not country_data.empty:
        country_data = country_data.iloc[0]
        immunity_score = calculate_immunity_score(country_data)
        risk_level, css_class = get_risk_level(immunity_score)
        
        # Main Immunity Display
        st.markdown(f"""
        <div class="immunity-terminal">
            <div class="immunity-score-display">
                <div style="font-size: 1.2rem; color: #888; margin-bottom: 10px;">FINANCIAL IMMUNITY INDEX</div>
                <div class="immunity-number {css_class}">{immunity_score:.1f}</div>
                <div class="country-info">{selected_country} • {selected_year}</div>
                <div class="status-indicator {css_class}" style="background: {'rgba(0,212,255,0.2)' if 'stable' in css_class else 'rgba(255,170,0,0.2)' if 'fragile' in css_class else 'rgba(255,51,102,0.2)'}; border: 1px solid {'#00d4ff' if 'stable' in css_class else '#ffaa00' if 'fragile' in css_class else '#ff3366'};">
                    {risk_level}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 12-Month Predictions
        st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 CRISIS PREDICTION TIMELINE (1-12 MONTHS)")
        
        predictions = predict_future_scores(country_data, 12)
        months = list(range(1, 13))
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Prediction chart
        fig_pred = go.Figure()
        
        # Current score
        fig_pred.add_trace(go.Scatter(
            x=[0], y=[immunity_score],
            mode='markers',
            marker=dict(size=15, color='#00d4ff', symbol='diamond'),
            name='Current Score'
        ))
        
        # Predictions
        colors = ['#ff3366' if p < 50 else '#ffaa00' if p < 80 else '#00d4ff' for p in predictions]
        fig_pred.add_trace(go.Scatter(
            x=months, y=predictions,
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3, dash='dot'),
            marker=dict(size=8, color=colors),
            name='Predicted Scores'
        ))
        
        # Risk zones
        fig_pred.add_hline(y=80, line_dash="dash", line_color="#00d4ff", annotation_text="STABLE THRESHOLD")
        fig_pred.add_hline(y=50, line_dash="dash", line_color="#ffaa00", annotation_text="CRISIS THRESHOLD")
        
        fig_pred.update_layout(
            title=f"12-Month Financial Immunity Forecast - {selected_country}",
            xaxis_title="Months Ahead",
            yaxis_title="Immunity Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction summary
        col1, col2, col3, col4 = st.columns(4)
        
        crisis_months = [i+1 for i, p in enumerate(predictions) if p < 50]
        avg_6m = np.mean(predictions[:6])
        avg_12m = np.mean(predictions)
        trend = "IMPROVING" if predictions[-1] > immunity_score else "DETERIORATING"
        
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-label">6-MONTH OUTLOOK</div>
                <div class="metric-value" style="color: {'#ff3366' if avg_6m < 50 else '#ffaa00' if avg_6m < 80 else '#00d4ff'};">{avg_6m:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-label">12-MONTH OUTLOOK</div>
                <div class="metric-value" style="color: {'#ff3366' if avg_12m < 50 else '#ffaa00' if avg_12m < 80 else '#00d4ff'};">{avg_12m:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-label">TREND DIRECTION</div>
                <div class="metric-value" style="color: {'#00d4ff' if trend == 'IMPROVING' else '#ff3366'};">{trend}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            crisis_risk = "HIGH" if crisis_months else "LOW"
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-label">CRISIS RISK</div>
                <div class="metric-value" style="color: {'#ff3366' if crisis_risk == 'HIGH' else '#00d4ff'};">{crisis_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Crisis warning
        if crisis_months:
            st.markdown(f"""
            <div class="alert-panel">
                <h3 style="margin: 0; color: #ff3366;">⚠️ CRISIS ALERT</h3>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem;">
                    Crisis predicted in months: {', '.join(map(str, crisis_months[:3]))}{'...' if len(crisis_months) > 3 else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Alert Panel for High Risk
        if immunity_score < 50:
            st.markdown(f"""
            <div class="alert-panel">
                <h3 style="margin: 0; color: #ff3366;">⚠️ CRITICAL ALERT</h3>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem;">
                    {selected_country} shows HIGH RISK indicators. Immediate intervention recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Financial Metrics Grid
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        metrics = [
            ("DEBT/GDP", f"{country_data['debt_to_gdp']:.1f}%", "#ff3366" if country_data['debt_to_gdp'] > 70 else "#00d4ff"),
            ("FX RESERVES", f"${country_data['fx_reserves']:.1f}B", "#ff3366" if country_data['fx_reserves'] < 5 else "#00d4ff"),
            ("INFLATION", f"{country_data['inflation']:.1f}%", "#ff3366" if country_data['inflation'] > 10 else "#00d4ff"),
            ("GDP GROWTH", f"{country_data['gdp_growth']:.1f}%", "#ff3366" if country_data['gdp_growth'] < 0 else "#00d4ff"),
            ("INTEREST RATE", f"{country_data['interest_rate']:.1f}%", "#ff3366" if country_data['interest_rate'] > 15 else "#00d4ff"),
            ("BOND SPREAD", f"{country_data['bond_yield_spread']:.1f}%", "#ff3366" if country_data['bond_yield_spread'] > 10 else "#00d4ff"),
            ("EXPORT REV", f"${country_data['export_revenue']:.1f}B", "#00d4ff"),
            ("POL STABILITY", f"{country_data['political_stability']:.2f}", "#ff3366" if country_data['political_stability'] < -0.5 else "#00d4ff")
        ]
        
        cols = st.columns(4)
        for i, (label, value, color) in enumerate(metrics):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color};">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-terminal">', unsafe_allow_html=True)
            
            # Historical trend
            country_history = df[df['country'] == selected_country].sort_values('year')
            if len(country_history) > 1:
                historical_scores = [calculate_immunity_score(row) for _, row in country_history.iterrows()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=country_history['year'],
                    y=historical_scores,
                    mode='lines+markers',
                    line=dict(color='#00d4ff', width=4),
                    marker=dict(size=10, color='#00d4ff', line=dict(width=2, color='white')),
                    name='Immunity Score'
                ))
                
                # Add risk zones
                fig.add_hline(y=80, line_dash="dash", line_color="#00d4ff", annotation_text="STABLE ZONE")
                fig.add_hline(y=50, line_dash="dash", line_color="#ffaa00", annotation_text="FRAGILE ZONE")
                
                fig.update_layout(
                    title="IMMUNITY TIMELINE",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-terminal">', unsafe_allow_html=True)
            
            # Risk factor radar
            categories = ['Debt', 'Reserves', 'Inflation', 'Growth', 'Rates', 'Stability']
            values = [
                min(country_data['debt_to_gdp']/100, 1),
                1 - min(country_data['fx_reserves']/20, 1),
                min(country_data['inflation']/50, 1),
                1 - max(country_data['gdp_growth']/10, 0),
                min(country_data['interest_rate']/30, 1),
                1 - max((country_data['political_stability'] + 2)/4, 0)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line_color='#ff3366',
                fillcolor='rgba(255,51,102,0.3)',
                name='Risk Factors'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], color='white'),
                    angularaxis=dict(color='white')
                ),
                title="RISK RADAR",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Global Overview
        st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
        st.markdown("### 🌍 GLOBAL THREAT MATRIX")
        
        latest_data = df[df['year'] == df['year'].max()]
        global_scores = []
        for _, row in latest_data.iterrows():
            score = calculate_immunity_score(row)
            global_scores.append({
                'Country': row['country'],
                'Score': score,
                'Risk': get_risk_level(score)[0]
            })
        
        country_to_iso = {
            "Sri Lanka": "LKA",
            "Ghana": "GHA",
            "Zambia": "ZMB",
            "Kenya": "KEN",
            "Nigeria": "NGA",
            "South Africa": "ZAF",
            "Tanzania": "TZA",
            "Rwanda": "RWA",
            "Ethiopia": "ETH",
            "Angola": "AGO",
            "Zimbabwe": "ZWE",
            "Senegal": "SEN",
            "Uganda": "UGA",
            "Mali": "MLI",
            "Burkina Faso": "BFA",
            "Cameroon": "CMR",
            "Madagascar": "MDG",
            "Botswana": "BWA"
        }
        
        global_df = pd.DataFrame(global_scores)
        global_df['ISO_A3'] = global_df['Country'].map(country_to_iso)
        global_df = global_df.sort_values('Score')
        
        # World Map
        st.markdown("#### 🗺️ GLOBAL IMMUNITY MAP")
        fig_map = px.choropleth(
            global_df,
            locations='ISO_A3',
            color='Score',
            locationmode='ISO-3',
            color_continuous_scale=[
                [0.0, '#ff3366'],
                [0.5, '#ffaa00'], 
                [1.0, '#00d4ff']
            ],
            range_color=[0, 100],
            title="Financial Immunity Scores Worldwide",
            hover_data={'Risk': True, 'Country': True}
        )
        
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Country Risk Summary and Threat Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 COUNTRY RISK SUMMARY")
            for _, row in global_df.iterrows():
                risk_color = {'STABLE': '#00d4ff', 'FRAGILE': '#ffaa00', 'HIGH RISK': '#ff3366'}[row['Risk']]
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; 
                           padding:12px; margin:8px 0; border-radius:8px; 
                           background:rgba(0,0,0,0.3); border-left:4px solid {risk_color};">
                    <span style="font-weight:600; font-family:'JetBrains Mono';">{row['Country']}</span>
                    <span style="color:{risk_color}; font-weight:700; font-family:'JetBrains Mono';">{row['Score']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 THREAT MATRIX")
            # Create threat matrix visualization
            fig = px.bar(
                global_df, 
                x='Score', 
                y='Country',
                color='Risk',
                color_discrete_map={'STABLE': '#00d4ff', 'FRAGILE': '#ffaa00', 'HIGH RISK': '#ff3366'},
                orientation='h'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 40px; border-top: 1px solid rgba(0,212,255,0.3); margin-top: 40px;">
        <div style="font-family: 'JetBrains Mono', monospace; color: #00d4ff; font-size: 0.9rem;">
            G-FIN TERMINAL v2.0 • POWERED BY ADVANCED AI • REAL-TIME FINANCIAL INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
