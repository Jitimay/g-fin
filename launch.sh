#!/bin/bash
echo "🚨 DCEWS - Debt Crisis Early Warning System"
echo "=========================================="
echo ""
echo "🔧 Checking system..."

# Check if model exists
if [ ! -f "model/debt_model.pkl" ]; then
    echo "📊 Training ML model..."
    python3 train_model.py
fi

echo "🚀 Launching dashboard..."
echo "📱 Open browser to: http://localhost:8501"
echo ""
streamlit run app.py
