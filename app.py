import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page config
st.set_page_config(
    page_title="G-FIN - Global Financial Immunity Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: #ffffff; }
    .stMetric { 
        background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
        padding: 20px !important; border-radius: 15px !important;
        border: 2px solid #4a90e2 !important;
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.3) !important;
        color: #212529 !important;
    }
    .immunity-score { font-size: 2em; font-weight: bold; text-align: center; }
    .stable { color: #28a745; }
    .fragile { color: #ffc107; }
    .high-risk { color: #dc3545; }
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

def calculate_immunity_score(model, features, country_data):
    """Calculate Financial Immunity Score"""
    X = country_data[features].values.reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    return 100 * (1 - prob)

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
    st.title("🛡️ G-FIN - Global Financial Immunity Network")
    st.markdown("**AI-Powered Financial Crisis Prediction System**")
    st.markdown("*Système immunitaire financier mondial qui prédit les crises et attribue un Financial Immunity Score*")
    st.markdown("**Score Range**: 80-100 (Stable) | 50-79 (Fragile) | 0-49 (High Risk)")
    
    # Load data and model
    df = load_data()
    model_data = load_model()
    model = model_data['model']
    features = model_data['features']
    
    # Sidebar
    st.sidebar.header("🎯 G-FIN Controls")
    
    # Country selection
    countries = df['country'].unique()
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    # Year selection
    years = sorted(df['year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
    
    # Get country data
    country_data = df[(df['country'] == selected_country) & (df['year'] == selected_year)]
    
    if not country_data.empty:
        country_data = country_data.iloc[0]
        
        # Calculate immunity score
        immunity_score = calculate_immunity_score(model, features, country_data)
        risk_level, css_class = get_risk_level(immunity_score)
        
        # Main dashboard
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="immunity-score {css_class}">
                Financial Immunity Score: {immunity_score:.1f}
            </div>
            <div style="text-align: center; font-size: 1.2em; margin-top: 10px;">
                {selected_country} ({selected_year}) - {risk_level}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🚨 Send Alert"):
                alert_msg = send_alert_simulation(selected_country, immunity_score)
                if alert_msg:
                    st.success(alert_msg)
                else:
                    st.info("No alert needed - country is stable")
        
        with col3:
            st.metric("Default Risk", f"{(100-immunity_score):.1f}%")
        
        # Key indicators
        st.subheader("📊 Key Financial Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Debt/GDP", f"{country_data['debt_to_gdp']:.1f}%")
            st.metric("Inflation", f"{country_data['inflation']:.1f}%")
        with col2:
            st.metric("FX Reserves", f"${country_data['fx_reserves']:.1f}B")
            st.metric("GDP Growth", f"{country_data['gdp_growth']:.1f}%")
        with col3:
            st.metric("Export Revenue", f"${country_data['export_revenue']:.1f}B")
            st.metric("Interest Rate", f"{country_data['interest_rate']:.1f}%")
        with col4:
            st.metric("Political Stability", f"{country_data['political_stability']:.2f}")
            st.metric("Bond Spread", f"{country_data['bond_yield_spread']:.1f}%")
        
        # Feature importance
        st.subheader("🔍 Model Explainability")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance in Crisis Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical trends
        st.subheader("📈 Historical Trends")
        country_history = df[df['country'] == selected_country].sort_values('year')
        
        if len(country_history) > 1:
            # Calculate historical immunity scores
            historical_scores = []
            for _, row in country_history.iterrows():
                score = calculate_immunity_score(model, features, row)
                historical_scores.append(score)
            
            country_history = country_history.copy()
            country_history['immunity_score'] = historical_scores
            
            fig = px.line(country_history, x='year', y='immunity_score',
                         title=f"{selected_country} - Financial Immunity Score Over Time")
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Stable Threshold")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Fragile Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Global overview
        st.subheader("🌍 Global Financial Immunity Overview")
        
        # Calculate scores for all countries in latest year
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        global_scores = []
        for _, row in latest_data.iterrows():
            score = calculate_immunity_score(model, features, row)
            global_scores.append({
                'Country': row['country'],
                'Immunity_Score': score,
                'Risk_Level': get_risk_level(score)[0]
            })
        
        global_df = pd.DataFrame(global_scores).sort_values('Immunity_Score', ascending=False)
        
        # World Map with Financial Immunity Scores
        st.subheader("🗺️ World Financial Immunity Map")
        
        # Create world map
        fig_map = px.choropleth(
            global_df,
            locations='Country',
            color='Immunity_Score',
            locationmode='country names',
            color_continuous_scale=[
                [0.0, '#dc3545'],    # Red for high risk (0-49)
                [0.5, '#ffc107'],    # Yellow for fragile (50-79)
                [1.0, '#28a745']     # Green for stable (80-100)
            ],
            range_color=[0, 100],
            title=f"Financial Immunity Scores by Country ({latest_year})",
            labels={'Immunity_Score': 'Financial Immunity Score'},
            hover_data={'Risk_Level': True}
        )
        
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            title_x=0.5,
            height=500
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Color mapping for bar chart
        color_map = {'STABLE': '#28a745', 'FRAGILE': '#ffc107', 'HIGH RISK': '#dc3545'}
        
        fig = px.bar(global_df, x='Country', y='Immunity_Score', color='Risk_Level',
                     color_discrete_map=color_map,
                     title=f"Global Financial Immunity Scores ({latest_year})")
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Stable Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Fragile Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.dataframe(global_df[['Country', 'Immunity_Score', 'Risk_Level']], use_container_width=True)
        
    else:
        st.error("No data available for selected country and year")

if __name__ == "__main__":
    main()
