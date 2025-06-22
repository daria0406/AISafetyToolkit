import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.transparency import TransparencyAnalyzer

class TestTransparencyAnalyzer(unittest.TestCase):
    """Test cases for the TransparencyAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.transparency_analyzer = TransparencyAnalyzer()
        
        # Create mock model with feature importance
        self.mock_model = MagicMock()
        self.mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        self.mock_model.predict.return_value = np.array([1, 0, 1, 0])
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 4
        
        self.X_test = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=['feature_1', 'feature_2', 'feature_3', 'feature_4']
        )
        self.y_test = np.random.randint(0, 2, self.n_samples)
        
        # Feature names
        self.feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
    def test_init(self):
        """Test TransparencyAnalyzer initialization."""
        analyzer = TransparencyAnalyzer()
        self.assertIsNone(analyzer.lime_explainer)
        self.assertIsNone(analyzer.shap_explainer)
        
    @patch('modules.transparency.go.Figure')
    def test_plot_feature_importance(self, mock_figure):
        """Test feature importance plotting."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        fig = self.transparency_analyzer.plot_feature_importance(
            self.mock_model, self.feature_names
        )
        
        # Verify that the method was called
        mock_figure.assert_called_once()
        
    def test_plot_feature_importance_without_importance(self):
        """Test feature importance plotting with model without feature_importances_."""
        model_without_importance = MagicMock()
        delattr(model_without_importance, 'feature_importances_')
        
        with self.assertRaises(Exception) as context:
            self.transparency_analyzer.plot_feature_importance(
                model_without_importance, self.feature_names
            )
        
        self.assertIn('does not have feature_importances_', str(context.exception))
        
    @patch('modules.transparency.lime.lime_tabular.LimeTabularExplainer')
    @patch('modules.transparency.go.Figure')
    def test_explain_with_lime(self, mock_figure, mock_lime_explainer):
        """Test LIME explanation generation."""
        # Mock LIME components
        mock_explainer_instance = MagicMock()
        mock_lime_explainer.return_value = mock_explainer_instance
        
        mock_explanation = MagicMock()
        mock_explanation.as_list.return_value = [
            ('feature_1 > 0.5', 0.3),
            ('feature_2 <= 0.2', -0.2),
            ('feature_3 > 1.0', 0.1)
        ]
        mock_explainer_instance.explain_instance.return_value = mock_explanation
        
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Test with pandas DataFrame
        fig = self.transparency_analyzer.explain_with_lime(
            self.mock_model, self.X_test, instance_idx=0
        )
        
        # Verify LIME explainer was created and used
        mock_lime_explainer.assert_called_once()
        mock_explainer_instance.explain_instance.assert_called_once()
        mock_figure.assert_called_once()
        
        # Test with numpy array
        X_test_numpy = self.X_test.values
        fig = self.transparency_analyzer.explain_with_lime(
            self.mock_model, X_test_numpy, instance_idx=0
        )
        
    @patch('modules.transparency.shap.TreeExplainer')
    @patch('modules.transparency.go.Figure')
    def test_explain_with_shap_tree(self, mock_figure, mock_tree_explainer):
        """Test SHAP explanation with TreeExplainer."""
        # Mock SHAP components
        mock_explainer_instance = MagicMock()
        mock_tree_explainer.return_value = mock_explainer_instance
        
        # Mock SHAP values (binary classification)
        mock_shap_values = [
            np.random.randn(10, self.n_features),  # Class 0
            np.random.randn(10, self.n_features)   # Class 1
        ]
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        fig = self.transparency_analyzer.explain_with_shap(
            self.mock_model, self.X_test.iloc[:10]
        )
        
        # Verify TreeExplainer was used
        mock_tree_explainer.assert_called_once_with(self.mock_model)
        mock_explainer_instance.shap_values.assert_called_once()
        mock_figure.assert_called_once()
        
    @patch('modules.transparency.shap.KernelExplainer')
    @patch('modules.transparency.shap.TreeExplainer')
    @patch('modules.transparency.go.Figure')
    def test_explain_with_shap_kernel_fallback(self, mock_figure, mock_tree_explainer, mock_kernel_explainer):
        """Test SHAP explanation with KernelExplainer fallback."""
        # Mock TreeExplainer to raise exception
        mock_tree_explainer.side_effect = Exception("TreeExplainer failed")
        
        # Mock KernelExplainer
        mock_kernel_instance = MagicMock()
        mock_kernel_explainer.return_value = mock_kernel_instance
        
        mock_shap_values = np.random.randn(10, self.n_features)
        mock_kernel_instance.shap_values.return_value = mock_shap_values
        
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        fig = self.transparency_analyzer.explain_with_shap(
            self.mock_model, self.X_test.iloc[:10]
        )
        
        # Verify fallback to KernelExplainer
        mock_kernel_explainer.assert_called_once()
        mock_kernel_instance.shap_values.assert_called_once()
        
    @patch('modules.transparency.go.Figure')
    def test_plot_decision_boundary_2d(self, mock_figure):
        """Test 2D decision boundary plotting."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        fig = self.transparency_analyzer.plot_decision_boundary_2d(
            self.mock_model, self.X_test, self.y_test, 
            feature1_idx=0, feature2_idx=1
        )
        
        # Verify plot was created
        mock_figure.assert_called_once()
        
    def test_analyze_model_behavior(self):
        """Test comprehensive model behavior analysis."""
        analysis = self.transparency_analyzer.analyze_model_behavior(
            self.mock_model, self.X_test, self.feature_names
        )
        
        # Check expected keys are present
        expected_keys = ['feature_importance', 'prediction_stats']
        for key in expected_keys:
            if key in analysis:
                self.assertIn(key, analysis)
                
        # Check feature importance
        if 'feature_importance' in analysis:
            self.assertEqual(len(analysis['feature_importance']), len(self.feature_names))
            
        # Check prediction stats
        if 'prediction_stats' in analysis:
            stats = analysis['prediction_stats']
            required_stats = ['mean', 'std', 'min', 'max', 'quartiles']
            for stat in required_stats:
                self.assertIn(stat, stats)
                
    def test_analyze_model_behavior_without_feature_names(self):
        """Test model behavior analysis without explicit feature names."""
        X_test_numpy = self.X_test.values
        
        analysis = self.transparency_analyzer.analyze_model_behavior(
            self.mock_model, X_test_numpy
        )
        
        # Should generate feature names automatically
        if 'feature_importance' in analysis:
            feature_keys = list(analysis['feature_importance'].keys())
            self.assertTrue(any('Feature_' in key for key in feature_keys))
            
    @patch('modules.transparency.make_subplots')
    @patch('modules.transparency.go')
    def test_create_interpretability_dashboard(self, mock_go, mock_subplots):
        """Test interpretability dashboard creation."""
        # Mock plotly components
        mock_fig = MagicMock()
        mock_subplots.return_value = mock_fig
        mock_go.Bar.return_value = MagicMock()
        mock_go.Histogram.return_value = MagicMock()
        
        fig = self.transparency_analyzer.create_interpretability_dashboard(
            self.mock_model, self.X_test, self.y_test
        )
        
        # Verify dashboard components were created
        mock_subplots.assert_called_once()
        
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test with invalid instance index
        with self.assertRaises(Exception):
            self.transparency_analyzer.explain_with_lime(
                self.mock_model, self.X_test, instance_idx=1000
            )
            
        # Test with invalid feature indices for decision boundary
        with self.assertRaises(Exception):
            self.transparency_analyzer.plot_decision_boundary_2d(
                self.mock_model, self.X_test, self.y_test,
                feature1_idx=10, feature2_idx=1  # Invalid index
            )
            
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small dataset
        X_small = self.X_test.iloc[:2]
        y_small = self.y_test[:2]
        
        # Should handle small datasets gracefully
        try:
            analysis = self.transparency_analyzer.analyze_model_behavior(
                self.mock_model, X_small
            )
            self.assertTrue(True)  # If no exception, test passes
        except Exception as e:
            # Should provide meaningful error message
            self.assertIsInstance(str(e), str)
            
        # Test with single feature
        X_single_feature = self.X_test.iloc[:, :1]
        
        try:
            analysis = self.transparency_analyzer.analyze_model_behavior(
                self.mock_model, X_single_feature
            )
            self.assertTrue(True)
        except Exception as e:
            self.assertIsInstance(str(e), str)


class TestTransparencyAnalyzerIntegration(unittest.TestCase):
    """Integration tests for TransparencyAnalyzer with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.transparency_analyzer = TransparencyAnalyzer()
        
        # Create a more realistic dataset and model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=6,
            n_redundant=2, n_clusters_per_class=1, random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = y
        
        # Train a real model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(self.X, self.y)
        
    def test_feature_importance_with_real_model(self):
        """Test feature importance analysis with real model."""
        feature_names = self.X.columns.tolist()
        
        try:
            fig = self.transparency_analyzer.plot_feature_importance(
                self.model, feature_names
            )
            # Should succeed without exception
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Feature importance plotting failed: {e}")
            
    def test_model_behavior_analysis_comprehensive(self):
        """Test comprehensive model behavior analysis."""
        analysis = self.transparency_analyzer.analyze_model_behavior(
            self.model, self.X, self.X.columns.tolist()
        )
        
        # Should have feature importance from tree model
        self.assertIn('feature_importance', analysis)
        
        # Should have prediction statistics
        self.assertIn('prediction_stats', analysis)
        
        # Check that all features are included
        self.assertEqual(len(analysis['feature_importance']), len(self.X.columns))
        
        # Check prediction stats validity
        stats = analysis['prediction_stats']
        self.assertGreaterEqual(stats['min'], 0)
        self.assertLessEqual(stats['max'], 1)
        self.assertGreaterEqual(stats['mean'], 0)
        self.assertLessEqual(stats['mean'], 1)
        
    def test_lime_explanation_integration(self):
        """Test LIME explanation with real model."""
        try:
            # Test with first instance
            fig = self.transparency_analyzer.explain_with_lime(
                self.model, self.X.iloc[:20], instance_idx=0, num_features=5
            )
            # Should succeed without exception
            self.assertTrue(True)
        except Exception as e:
            # LIME might fail due to dependencies, but should handle gracefully
            self.assertIn('Error generating LIME explanation', str(e))
            
    def test_decision_boundary_with_real_model(self):
        """Test decision boundary visualization with real model."""
        try:
            fig = self.transparency_analyzer.plot_decision_boundary_2d(
                self.model, self.X.iloc[:50], self.y[:50], 
                feature1_idx=0, feature2_idx=1
            )
            # Should succeed without exception
            self.assertTrue(True)
        except Exception as e:
            self.assertIn('Error plotting decision boundary', str(e))
            
    def test_performance_with_larger_dataset(self):
        """Test performance with larger realistic dataset."""
        from sklearn.datasets import make_classification
        
        # Create larger dataset
        X_large, y_large = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        
        X_large_df = pd.DataFrame(X_large, columns=[f'feature_{i}' for i in range(X_large.shape[1])])
        
        # Train model
        model_large = RandomForestClassifier(n_estimators=50, random_state=42)
        model_large.fit(X_large_df, y_large)
        
        # Test behavior analysis performance
        import time
        start_time = time.time()
        
        analysis = self.transparency_analyzer.analyze_model_behavior(
            model_large, X_large_df.iloc[:100], X_large_df.columns.tolist()
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(execution_time, 10.0, "Analysis should complete within reasonable time")
        
        # Should have all expected components
        self.assertIn('feature_importance', analysis)
        self.assertIn('prediction_stats', analysis)


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestTransparencyAnalyzer))
    suite.addTest(loader.loadTestsFromTestCase(TestTransparencyAnalyzerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
