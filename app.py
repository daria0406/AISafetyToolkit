import streamlit as st
import pandas as pd
import numpy as np
from modules.bias_detection import BiasDetector
from modules.fairness_evaluation import FairnessEvaluator
from modules.transparency import TransparencyAnalyzer
from modules.data_loader import DataLoader
from modules.utils import load_sample_model

# Configure page
st.set_page_config(
    page_title="AI Safety Toolkit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    st.title("🛡️ AI Safety Toolkit")
    st.markdown("""
    A comprehensive toolkit for AI safety analysis including bias detection, 
    fairness evaluation, and model transparency.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a module:",
        ["Overview", "Bias Detection", "Fairness Evaluation", "Model Transparency"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Bias Detection":
        show_bias_detection()
    elif page == "Fairness Evaluation":
        show_fairness_evaluation()
    elif page == "Model Transparency":
        show_transparency()

def show_overview():
    st.header("📊 Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Bias Detection")
        st.write("""
        Analyze ML models for demographic bias and statistical disparities.
        - Disparate Impact Analysis
        - Statistical Parity
        - Equalized Odds
        """)
    
    with col2:
        st.subheader("⚖️ Fairness Evaluation")
        st.write("""
        Interactive dashboard for fairness metrics visualization.
        - Group Fairness Metrics
        - Individual Fairness Analysis
        - Comparative Visualizations
        """)
    
    with col3:
        st.subheader("🔍 Model Transparency")
        st.write("""
        Explain model decisions using interpretability techniques.
        - LIME Explanations
        - SHAP Values
        - Feature Importance
        """)
    
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.write("Select a module from the sidebar to begin your AI safety analysis.")

def show_bias_detection():
    st.header("🎯 Bias Detection Module")
    
    # Data loading section
    st.subheader("1. Load Dataset")
    data_source = st.selectbox(
        "Choose data source:",
        ["Adult Income Dataset", "German Credit Dataset", "Upload Custom Dataset"]
    )
    
    data_loader = DataLoader()
    
    if data_source == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("Data loaded successfully!")
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        else:
            st.info("Please upload a CSV file to continue.")
            return
    else:
        try:
            if data_source == "Adult Income Dataset":
                data = data_loader.load_adult_dataset()
            else:  # German Credit Dataset
                data = data_loader.load_german_credit_dataset()
            st.success(f"{data_source} loaded successfully!")
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading {data_source}: {str(e)}")
            return
    
    if st.session_state.data_loaded:
        st.subheader("2. Dataset Overview")
        st.write(f"Shape: {data.shape}")
        st.dataframe(data.head())
        
        # Model training section
        st.subheader("3. Train Model")
        if st.button("Train Classification Model"):
            try:
                model, X_test, y_test, protected_attr = load_sample_model(data, data_source)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.protected_attr = protected_attr
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
        
        # Bias detection analysis
        if st.session_state.model_trained:
            st.subheader("4. Bias Detection Analysis")
            
            bias_detector = BiasDetector()
            
            try:
                # Get predictions
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                
                # Calculate bias metrics
                disparate_impact = bias_detector.calculate_disparate_impact(
                    y_pred, st.session_state.protected_attr
                )
                
                statistical_parity = bias_detector.calculate_statistical_parity(
                    y_pred, st.session_state.protected_attr
                )
                
                equalized_odds = bias_detector.calculate_equalized_odds(
                    st.session_state.y_test, y_pred, st.session_state.protected_attr
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Disparate Impact", f"{disparate_impact:.3f}")
                    if disparate_impact < 0.8:
                        st.error("⚠️ Potential bias detected!")
                    else:
                        st.success("✅ Within acceptable range")
                
                with col2:
                    st.metric("Statistical Parity Diff", f"{statistical_parity:.3f}")
                    if abs(statistical_parity) > 0.1:
                        st.error("⚠️ Potential bias detected!")
                    else:
                        st.success("✅ Within acceptable range")
                
                with col3:
                    st.metric("Equalized Odds Diff", f"{equalized_odds:.3f}")
                    if abs(equalized_odds) > 0.1:
                        st.error("⚠️ Potential bias detected!")
                    else:
                        st.success("✅ Within acceptable range")
                
                # Visualization
                fig = bias_detector.plot_bias_metrics(
                    y_pred, st.session_state.protected_attr, st.session_state.y_test
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in bias detection: {str(e)}")

def show_fairness_evaluation():
    st.header("⚖️ Fairness Evaluation Module")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Please train a model in the Bias Detection module first.")
        return
    
    fairness_evaluator = FairnessEvaluator()
    
    try:
        # Get predictions
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        y_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
        
        st.subheader("Fairness Metrics Dashboard")
        
        # Calculate comprehensive fairness metrics
        metrics = fairness_evaluator.calculate_all_metrics(
            st.session_state.y_test,
            y_pred,
            y_proba,
            st.session_state.protected_attr
        )
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Group Fairness Metrics")
            for metric, value in metrics['group_fairness'].items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
        
        with col2:
            st.subheader("Predictive Parity Metrics")
            for metric, value in metrics['predictive_parity'].items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
        
        # Fairness visualization
        st.subheader("Fairness Visualization")
        fig = fairness_evaluator.create_fairness_dashboard(
            st.session_state.y_test,
            y_pred,
            y_proba,
            st.session_state.protected_attr
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation system
        st.subheader("Fairness Recommendations")
        recommendations = fairness_evaluator.generate_recommendations(metrics)
        for rec in recommendations:
            st.info(rec)
            
    except Exception as e:
        st.error(f"Error in fairness evaluation: {str(e)}")

def show_transparency():
    st.header("🔍 Model Transparency Module")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Please train a model in the Bias Detection module first.")
        return
    
    transparency_analyzer = TransparencyAnalyzer()
    
    try:
        st.subheader("Model Interpretability Analysis")
        
        # Feature importance
        st.subheader("1. Feature Importance")
        if hasattr(st.session_state.model, 'feature_importances_'):
            feature_names = st.session_state.X_test.columns if hasattr(st.session_state.X_test, 'columns') else [f'Feature_{i}' for i in range(st.session_state.X_test.shape[1])]
            importance_fig = transparency_analyzer.plot_feature_importance(
                st.session_state.model, feature_names
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # LIME explanation
        st.subheader("2. LIME Explanations")
        instance_idx = st.slider(
            "Select instance to explain:",
            0, len(st.session_state.X_test) - 1, 0
        )
        
        if st.button("Generate LIME Explanation"):
            try:
                lime_fig = transparency_analyzer.explain_with_lime(
                    st.session_state.model,
                    st.session_state.X_test,
                    instance_idx
                )
                st.plotly_chart(lime_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
        
        # SHAP explanation
        st.subheader("3. SHAP Explanations")
        if st.button("Generate SHAP Analysis"):
            try:
                shap_fig = transparency_analyzer.explain_with_shap(
                    st.session_state.model,
                    st.session_state.X_test
                )
                st.plotly_chart(shap_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {str(e)}")
        
        # Decision boundary visualization (for 2D case)
        st.subheader("4. Model Behavior Analysis")
        if st.session_state.X_test.shape[1] >= 2:
            feature1 = st.selectbox("Select first feature:", range(st.session_state.X_test.shape[1]), 0)
            feature2 = st.selectbox("Select second feature:", range(st.session_state.X_test.shape[1]), 1)
            
            if st.button("Visualize Decision Boundary"):
                try:
                    boundary_fig = transparency_analyzer.plot_decision_boundary_2d(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        feature1, feature2
                    )
                    st.plotly_chart(boundary_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error visualizing decision boundary: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error in transparency analysis: {str(e)}")

if __name__ == "__main__":
    main()
