# AI Safety Toolkit

## Overview

This is a comprehensive AI Safety Toolkit built with Streamlit that provides bias detection, fairness evaluation, and model transparency analysis for machine learning models. The application is designed to help data scientists and ML engineers identify and address potential fairness issues in their models through interactive analysis tools.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page navigation
- **UI Components**: Interactive dashboards with Plotly visualizations for bias metrics, fairness evaluation, and model interpretability
- **Layout**: Sidebar navigation with wide layout configuration for comprehensive data visualization

### Backend Architecture
- **Modular Design**: Separated into distinct modules for different analysis types:
  - `bias_detection.py`: Implements bias detection algorithms (disparate impact, statistical parity)
  - `fairness_evaluation.py`: Comprehensive fairness metrics evaluation
  - `transparency.py`: Model interpretability using LIME, SHAP, and feature importance
  - `data_loader.py`: Handles dataset loading and preprocessing
  - `utils.py`: Utility functions for model training and data preparation

### Data Processing
- **Data Sources**: Support for OpenML datasets (Adult Income, German Credit) with fallback to synthetic data generation
- **Preprocessing**: Automated encoding of categorical variables, scaling, and missing value handling
- **Protected Attributes**: Automatic detection and handling of sensitive attributes for fairness analysis

## Key Components

### Bias Detection Module
- **Disparate Impact Analysis**: Calculates ratio of positive prediction rates between groups
- **Statistical Parity**: Evaluates equal treatment across demographic groups
- **Equalized Odds**: Assesses fairness in error rates between protected groups
- **Visualization**: Interactive plots showing bias metrics and group comparisons

### Fairness Evaluation Dashboard
- **Group Fairness Metrics**: Demographic parity, equalized odds, and predictive parity
- **Individual Fairness**: Analysis of similar treatment for similar individuals
- **Calibration Analysis**: Ensures prediction probabilities reflect true outcomes
- **Comparative Visualizations**: ROC curves, precision-recall curves, and calibration plots

### Model Transparency Module
- **LIME Integration**: Local interpretable model-agnostic explanations for individual predictions
- **SHAP Values**: Unified approach to feature importance and model interpretability
- **Feature Importance**: Built-in and permutation-based importance analysis
- **Decision Boundary Visualization**: 2D visualization of model behavior

### Data Loading System
- **Dataset Support**: Adult Income dataset from OpenML with preprocessing pipeline
- **Synthetic Data**: Fallback generation of synthetic datasets with bias patterns
- **Preprocessing Pipeline**: Automated handling of categorical encoding, scaling, and data splitting

## Data Flow

1. **Data Ingestion**: Users can load predefined datasets (Adult Income) or upload custom data
2. **Preprocessing**: Automatic detection of target variables and protected attributes, encoding, and scaling
3. **Model Training**: Quick model training using Random Forest or Logistic Regression for demonstration
4. **Analysis Pipeline**: 
   - Bias detection calculates disparate impact and group-based metrics
   - Fairness evaluation computes comprehensive fairness metrics across groups
   - Transparency analysis generates LIME/SHAP explanations and feature importance
5. **Visualization**: Interactive Plotly charts display results with detailed breakdowns

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Model training, evaluation, and preprocessing
- **pandas/numpy**: Data manipulation and numerical computations

### Visualization
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive plotting and dashboard visualizations
- **matplotlib/seaborn**: Additional plotting capabilities

### Interpretability
- **LIME**: Local interpretable model-agnostic explanations
- **SHAP**: SHapley Additive exPlanations for model interpretability

### Data Sources
- **OpenML**: Access to standardized fairness evaluation datasets

## Deployment Strategy

- **Platform**: Replit with autoscale deployment target
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: Streamlit server running on port 5000
- **Dependencies**: UV package manager with PyTorch CPU index for ML libraries
- **System Packages**: Cairo, FFmpeg, FreeType for visualization and media processing

## Changelog

- June 21, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.