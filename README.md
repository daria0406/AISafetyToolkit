# AI Safety Toolkit

A comprehensive AI Safety Toolkit with bias detection, fairness evaluation, and model transparency modules built using Streamlit and Python ML libraries.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Bias Detection Module
- **Disparate Impact Analysis**: Measure discrimination through statistical ratios
- **Statistical Parity**: Evaluate equal treatment across demographic groups
- **Equalized Odds**: Assess fairness in error rates between groups
- **Interactive Visualizations**: Comprehensive bias metrics dashboard

### Fairness Evaluation Dashboard
- **Group Fairness Metrics**: Demographic parity, equalized odds, and more
- **Individual Fairness Analysis**: Assess similar treatment for similar individuals
- **Predictive Parity**: Evaluate accuracy consistency across groups
- **Calibration Analysis**: Ensure prediction probabilities reflect true outcomes
- **Comparative Visualizations**: ROC curves, precision-recall, and calibration plots

###  Model Transparency Module
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **SHAP Values**: Unified approach to model interpretability
- **Feature Importance**: Built-in and permutation-based importance analysis
- **Decision Boundary Visualization**: 2D model behavior visualization
- **Interactive Dashboard**: Comprehensive interpretability analysis

## Requirements

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- plotly
- LIME
- SHAP
- seaborn
- matplotlib

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-safety-toolkit.git
cd ai-safety-toolkit
```

2. **Install libraries:**
```bash
pip install -r requirements.txt
