# 🚨 Debt Crisis Early Warning System (DCEWS)

AI-powered platform that predicts sovereign debt default risk 6-12 months in advance for African countries.

## 🎯 Features

- **ML Prediction Model**: Random Forest classifier predicting default probability
- **Interactive Dashboard**: Real-time risk visualization with Streamlit
- **Risk Mapping**: Color-coded world map showing country risk levels
- **Explainable AI**: Feature importance and risk factor analysis
- **Historical Trends**: Time-series analysis of key indicators

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo>
cd dcews

# Generate data and train model
python3 generate_data.py
python3 train_model.py

# Launch dashboard
streamlit run app.py
```

## 📊 Current Results (2023 Predictions)

```
🔴 HIGH RISK:  Ghana (99%), Zambia (97%), Ethiopia (92%)
🟡 WARNING:    Kenya (61%)
🟢 STABLE:     6 other countries
```

## 🔧 Technical Stack

- **Backend**: Python, scikit-learn
- **Frontend**: Streamlit, Plotly
- **Data**: Synthetic macroeconomic indicators for 10 African countries

## 📈 Key Indicators

- Debt-to-GDP ratio
- Foreign exchange reserves  
- Inflation rate
- Interest rates
- GDP growth
- Export revenues
- Budget balance
- Political stability
- Bond yield spreads

## 🎨 Dashboard Features

1. **Risk Overview**: Summary metrics and alerts
2. **Interactive Map**: Geographic risk visualization  
3. **Country Analysis**: Detailed risk breakdown
4. **Historical Trends**: Time-series charts
5. **Risk Table**: Sortable country rankings
6. **Feature Importance**: Model explainability

## 📁 Project Structure

```
dcews/
├── data/debt_data.csv      # Synthetic country data
├── model/debt_model.pkl    # Trained ML model
├── app.py                  # Streamlit dashboard
├── train_model.py          # Model training script
├── demo.py                 # CLI demo
├── launch.sh               # Auto-launch script
└── README.md               # This file
```

## 🏆 Hackathon Ready

- ✅ Runs completely offline
- ✅ No external API dependencies  
- ✅ Synthetic data included
- ✅ Professional UI design
- ✅ Explainable AI results
- ✅ One-command launch

Built for hackathons and demos - works out of the box!
