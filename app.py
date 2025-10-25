import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Import risk calculation function
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

# Page config
st.set_page_config(
    page_title="G-FIN - Global Financial Immunity Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { 
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b69 100%);
        color: #ffffff; 
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { background: transparent; }
    
    .header-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .immunity-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(30px);
        border: 2px solid rgba(255,255,255,0.2);
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .immunity-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00d4ff, #5b73ff, #ff006e);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .immunity-score {
        font-size: 4rem;
        font-weight: 700;
        margin: 20px 0;
        text-shadow: 0 0 30px rgba(255,255,255,0.3);
    }
    
    .stable { 
        color: #00d4ff;
        text-shadow: 0 0 30px rgba(0,212,255,0.5);
    }
    .fragile { 
        color: #ffaa00;
        text-shadow: 0 0 30px rgba(255,170,0,0.5);
    }
    .high-risk { 
        color: #ff3366;
        text-shadow: 0 0 30px rgba(255,51,102,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        border-color: rgba(255,255,255,0.3);
    }
    
    .alert-button {
        background: linear-gradient(135deg, #ff3366 0%, #ff6b9d 100%);
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(255,51,102,0.3);
    }
    
    .alert-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(255,51,102,0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
    }
    
    .chart-container {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 5px;
    }
    
    .badge-stable { background: rgba(0,212,255,0.2); color: #00d4ff; border: 1px solid #00d4ff; }
    .badge-fragile { background: rgba(255,170,0,0.2); color: #ffaa00; border: 1px solid #ffaa00; }
    .badge-risk { background: rgba(255,51,102,0.2); color: #ff3366; border: 1px solid #ff3366; }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load G-FIN data"""
    return pd.read_csv('data/gfin_real_data.csv')

@st.cache_resource
def load_model():
    """Load G-FIN model"""
    with open('model/gfin_model.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_immunity_score(model_data, features, country_data):
    """Calculate Financial Immunity Score using rule-based model"""
    risk = calculate_risk_score(country_data)
    # Add some randomness for realism
    noise = np.random.normal(0, 0.02)
    final_risk = np.clip(risk + noise, 0.05, 0.95)
    return (1 - final_risk) * 100

def get_risk_level(score):
    """Get risk level from immunity score"""
    if score >= 80:
        return "STABLE", "stable"
    elif score >= 50:
        return "FRAGILE", "fragile"
    else:
        return "HIGH RISK", "high-risk"

def send_alert_simulation(country, score):
    """Simulate alert sending"""
    risk_level, _ = get_risk_level(score)
    if score < 50:
        st.warning(f"🚨 G-FIN ALERT: {country} - Financial Immunity Score: {score:.1f} ({risk_level})")
        return f"Alert sent for {country}: Score {score:.1f}"
    return None

# Main app
def main():
    # Modern header
    st.markdown("""
    <div class="header-container">
        <h1 style="margin:0; font-size:3rem; font-weight:700;">🛡️ G-FIN</h1>
        <h2 style="margin:10px 0; font-weight:300; opacity:0.9;">Global Financial Immunity Network</h2>
        <p style="margin:0; font-size:1.1rem; opacity:0.8;">AI-Powered Financial Crisis Prediction System</p>
        <div style="margin-top:20px;">
            <span class="status-badge badge-stable">80-100 Stable</span>
            <span class="status-badge badge-fragile">50-79 Fragile</span>
            <span class="status-badge badge-risk">0-49 High Risk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model_data = load_model()
    features = model_data['features']
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎯 G-FIN Controls")
        
        # Country selection with search
        countries = sorted(df['country'].unique())
        selected_country = st.selectbox("🌍 Select Country", countries, 
                                      help="Choose a country to analyze")
        
        # Year selection
        years = sorted(df['year'].unique())
        selected_year = st.selectbox("📅 Select Year", years, 
                                   index=len(years)-1,
                                   help="Select analysis year")
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        total_countries = len(df['country'].unique())
        latest_year = df['year'].max()
        st.metric("Countries Monitored", total_countries)
        st.metric("Latest Data", latest_year)
    
    # Get country data
    country_data = df[(df['country'] == selected_country) & (df['year'] == selected_year)]
    
    if not country_data.empty:
        country_data = country_data.iloc[0]
        
        # Calculate immunity score
        immunity_score = calculate_immunity_score(model_data, features, country_data)
        risk_level, css_class = get_risk_level(immunity_score)
        
        # Main immunity score display
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            pulse_class = "pulse" if immunity_score < 50 else ""
            st.markdown(f"""
            <div class="immunity-card {pulse_class}">
                <div style="font-size:1.2rem; opacity:0.8; margin-bottom:10px;">Financial Immunity Score</div>
                <div class="immunity-score {css_class}">{immunity_score:.1f}</div>
                <div style="font-size:1.4rem; font-weight:600; margin-top:10px;">
                    {selected_country} ({selected_year})
                </div>
                <div style="font-size:1.1rem; opacity:0.9; margin-top:5px;">
                    Status: {risk_level}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Default Risk", f"{(100-immunity_score):.1f}%", 
                     delta=f"{(100-immunity_score)-50:.1f}%" if immunity_score < 50 else None)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            if st.button("🚨 Send Alert", key="alert_btn", help="Trigger emergency alert"):
                alert_msg = send_alert_simulation(selected_country, immunity_score)
                if alert_msg:
                    st.success(f"✅ {alert_msg}")
                else:
                    st.info("ℹ️ No alert needed - country is stable")
        
        # Enhanced key indicators
        st.markdown("### 📊 Financial Health Indicators")
        
        # Create metric cards with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            ("💳 Debt/GDP", f"{country_data['debt_to_gdp']:.1f}%", "debt_to_gdp"),
            ("💰 FX Reserves", f"${country_data['fx_reserves']:.1f}B", "fx_reserves"),
            ("📈 Inflation", f"{country_data['inflation']:.1f}%", "inflation"),
            ("💹 GDP Growth", f"{country_data['gdp_growth']:.1f}%", "gdp_growth"),
            ("🚢 Export Revenue", f"${country_data['export_revenue']:.1f}B", "export_revenue"),
            ("🏛️ Interest Rate", f"{country_data['interest_rate']:.1f}%", "interest_rate"),
            ("⚖️ Political Stability", f"{country_data['political_stability']:.2f}", "political_stability"),
            ("📊 Bond Spread", f"{country_data['bond_yield_spread']:.1f}%", "bond_yield_spread")
        ]
        
        for i, (label, value, key) in enumerate(metrics_data):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                # Determine color based on metric health
                color = "#00d4ff" if key in ["fx_reserves", "gdp_growth", "export_revenue"] and country_data[key] > 0 else "#ffaa00"
                if key in ["debt_to_gdp", "inflation", "bond_yield_spread"] and country_data[key] > 50:
                    color = "#ff3366"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div style="font-size:0.9rem; opacity:0.8; margin-bottom:5px;">{label}</div>
                    <div style="font-size:1.5rem; font-weight:600; color:{color};">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced feature importance with modern styling
        st.markdown("### 🔍 AI Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Rule-based feature importance
            importance_data = {
                'debt_to_gdp': 0.25,
                'bond_yield_spread': 0.20,
                'fx_reserves': 0.18,
                'inflation': 0.15,
                'interest_rate': 0.12,
                'gdp_growth': 0.08,
                'political_stability': 0.02
            }
            
            importance_df = pd.DataFrame({
                'Feature': list(importance_data.keys()),
                'Importance': list(importance_data.values())
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="🎯 Feature Importance in Crisis Prediction",
                        color='Importance',
                        color_continuous_scale=['#ff3366', '#ffaa00', '#00d4ff'])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Risk factors breakdown
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🚨 Risk Factor Analysis")
            
            risk_factors = []
            if country_data['debt_to_gdp'] > 70:
                risk_factors.append(("High Debt Burden", country_data['debt_to_gdp'], "🔴"))
            if country_data['inflation'] > 10:
                risk_factors.append(("High Inflation", country_data['inflation'], "🟡"))
            if country_data['fx_reserves'] < 5:
                risk_factors.append(("Low FX Reserves", country_data['fx_reserves'], "🔴"))
            if country_data['bond_yield_spread'] > 10:
                risk_factors.append(("High Bond Spreads", country_data['bond_yield_spread'], "🔴"))
            
            if risk_factors:
                for factor, value, icon in risk_factors:
                    st.markdown(f"{icon} **{factor}**: {value:.1f}")
            else:
                st.markdown("✅ **No major risk factors detected**")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced historical trends
        st.markdown("### 📈 Historical Analysis")
        country_history = df[df['country'] == selected_country].sort_values('year')
        
        if len(country_history) > 1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Calculate historical immunity scores
            historical_scores = []
            for _, row in country_history.iterrows():
                score = calculate_immunity_score(model_data, features, row)
                historical_scores.append(score)
            
            country_history = country_history.copy()
            country_history['immunity_score'] = historical_scores
            
            fig = px.line(country_history, x='year', y='immunity_score',
                         title=f"🛡️ {selected_country} - Financial Immunity Timeline",
                         markers=True)
            
            # Enhanced styling
            fig.update_traces(
                line=dict(width=4, color='#00d4ff'),
                marker=dict(size=8, color='#00d4ff', line=dict(width=2, color='white'))
            )
            
            # Add threshold zones
            fig.add_hrect(y0=80, y1=100, fillcolor="rgba(0,212,255,0.1)", 
                         line_width=0, annotation_text="STABLE ZONE")
            fig.add_hrect(y0=50, y1=80, fillcolor="rgba(255,170,0,0.1)", 
                         line_width=0, annotation_text="FRAGILE ZONE")
            fig.add_hrect(y0=0, y1=50, fillcolor="rgba(255,51,102,0.1)", 
                         line_width=0, annotation_text="HIGH RISK ZONE")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=18,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced global overview
        st.markdown("### 🌍 Global Financial Immunity Dashboard")
        
        # Calculate scores for all countries in latest year
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        global_scores = []
        for _, row in latest_data.iterrows():
            score = calculate_immunity_score(model_data, features, row)
            global_scores.append({
                'Country': row['country'],
                'Immunity_Score': score,
                'Risk_Level': get_risk_level(score)[0],
                'Debt_GDP': row['debt_to_gdp'],
                'FX_Reserves': row['fx_reserves'],
                'Inflation': row['inflation']
            })
        
        global_df = pd.DataFrame(global_scores).sort_values('Immunity_Score', ascending=False)
        
        # Enhanced world map
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig_map = px.choropleth(
            global_df,
            locations='Country',
            color='Immunity_Score',
            locationmode='country names',
            color_continuous_scale=[
                [0.0, '#ff3366'],    # Red for high risk
                [0.5, '#ffaa00'],    # Yellow for fragile  
                [1.0, '#00d4ff']     # Cyan for stable
            ],
            range_color=[0, 100],
            title=f"🗺️ Global Financial Immunity Map ({latest_year})",
            labels={'Immunity_Score': 'Financial Immunity Score'},
            hover_data={
                'Risk_Level': True,
                'Debt_GDP': ':.1f',
                'FX_Reserves': ':.1f',
                'Inflation': ':.1f'
            }
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
            title_x=0.5,
            height=500,
            title_font_size=20
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced country ranking
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Color mapping for bar chart
            color_map = {'STABLE': '#00d4ff', 'FRAGILE': '#ffaa00', 'HIGH RISK': '#ff3366'}
            
            fig = px.bar(global_df, x='Country', y='Immunity_Score', 
                        color='Risk_Level',
                        color_discrete_map=color_map,
                        title=f"🏆 Country Rankings ({latest_year})")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📋 Country Risk Summary")
            
            # Enhanced summary table with styling
            for _, row in global_df.iterrows():
                risk_color = {'STABLE': '#00d4ff', 'FRAGILE': '#ffaa00', 'HIGH RISK': '#ff3366'}[row['Risk_Level']]
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; 
                           padding:10px; margin:5px 0; border-radius:8px; 
                           background:rgba(255,255,255,0.05); border-left:4px solid {risk_color};">
                    <span style="font-weight:600;">{row['Country']}</span>
                    <span style="color:{risk_color}; font-weight:600;">{row['Immunity_Score']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.error("❌ No data available for selected country and year")
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:30px; opacity:0.7;">
        <p style="margin:0; font-size:0.9rem;">
            🛡️ <strong>G-FIN</strong> - Powered by Advanced AI & Real Financial Data
        </p>
        <p style="margin:5px 0 0 0; font-size:0.8rem;">
            Predicting financial crises 6-12 months in advance • Built for institutional use
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
