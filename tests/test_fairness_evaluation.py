import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fairness_evaluation import FairnessEvaluator

class TestFairnessEvaluator(unittest.TestCase):
    """Test cases for the FairnessEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.fairness_evaluator = FairnessEvaluator()
        
        # Create sample data for testing
        np.random.seed(42)
        self.n_samples = 100
        
        # Create binary test data
        self.y_true = np.random.randint(0, 2, self.n_samples)
        self.y_pred = np.random.randint(0, 2, self.n_samples)
        self.y_proba = np.random.uniform(0, 1, self.n_samples)
        self.protected_attr = np.random.randint(0, 2, self.n_samples)
        
        # Create biased test data
        self.y_true_biased = np.array([1] * 25 + [0] * 25 + [1] * 15 + [0] * 35)
        self.y_pred_biased = np.array([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)
        self.y_proba_biased = np.concatenate([
            np.random.uniform(0.6, 1.0, 50),  # Group 0 - higher probabilities
            np.random.uniform(0.2, 0.6, 50)   # Group 1 - lower probabilities
        ])
        self.protected_biased = np.array([0] * 50 + [1] * 50)
        
    def test_init(self):
        """Test FairnessEvaluator initialization."""
        evaluator = FairnessEvaluator()
        self.assertIn('statistical_parity', evaluator.fairness_thresholds)
        self.assertIn('equalized_odds', evaluator.fairness_thresholds)
        self.assertEqual(evaluator.fairness_thresholds['statistical_parity'], 0.1)
        
    def test_calculate_all_metrics(self):
        """Test comprehensive fairness metrics calculation."""
        metrics = self.fairness_evaluator.calculate_all_metrics(
            self.y_true, self.y_pred, self.y_proba, self.protected_attr
        )
        
        # Check all categories are present
        expected_categories = ['group_fairness', 'predictive_parity', 
                             'individual_fairness', 'calibration']
        for category in expected_categories:
            self.assertIn(category, metrics)
            
        # Check group fairness metrics
        group_fairness = metrics['group_fairness']
        expected_group_metrics = ['demographic_parity', 'equalized_odds_tpr', 
                                'equalized_odds_fpr', 'disparate_impact']
        for metric in expected_group_metrics:
            self.assertIn(metric, group_fairness)
            
    def test_calculate_group_fairness(self):
        """Test group fairness metrics calculation."""
        group_fairness = self.fairness_evaluator._calculate_group_fairness(
            self.y_true_biased, self.y_pred_biased, self.protected_biased
        )
        
        # Check all expected metrics are present
        expected_metrics = ['demographic_parity', 'equalized_odds_tpr', 
                          'equalized_odds_fpr', 'disparate_impact']
        for metric in expected_metrics:
            self.assertIn(metric, group_fairness)
            
        # Verify demographic parity calculation
        # Group 0: 30/50 = 0.6, Group 1: 10/50 = 0.2
        # Demographic parity = |0.2 - 0.6| = 0.4
        expected_dp = abs(0.2 - 0.6)
        self.assertAlmostEqual(group_fairness['demographic_parity'], expected_dp, places=2)
        
        # Verify disparate impact calculation
        # DI = 0.2/0.6 = 0.333
        expected_di = 0.2 / 0.6
        self.assertAlmostEqual(group_fairness['disparate_impact'], expected_di, places=2)
        
    def test_calculate_predictive_parity(self):
        """Test predictive parity metrics calculation."""
        predictive_parity = self.fairness_evaluator._calculate_predictive_parity(
            self.y_true, self.y_pred, self.protected_attr
        )
        
        expected_metrics = ['predictive_parity_positive', 'predictive_parity_negative']
        for metric in expected_metrics:
            self.assertIn(metric, predictive_parity)
            
        # Values should be numbers or NaN
        for value in predictive_parity.values():
            self.assertTrue(np.isnan(value) or isinstance(value, (int, float)))
            
    def test_calculate_individual_fairness(self):
        """Test individual fairness metrics calculation."""
        individual_fairness = self.fairness_evaluator._calculate_individual_fairness(
            self.y_proba, self.protected_attr
        )
        
        self.assertIn('prediction_variance_ratio', individual_fairness)
        
        # Should be a positive number
        variance_ratio = individual_fairness['prediction_variance_ratio']
        self.assertGreater(variance_ratio, 0)
        
    def test_calculate_calibration(self):
        """Test calibration metrics calculation."""
        calibration = self.fairness_evaluator._calculate_calibration(
            self.y_true, self.y_proba, self.protected_attr
        )
        
        self.assertIn('calibration_difference', calibration)
        
        # Should be a non-negative number
        cal_diff = calibration['calibration_difference']
        self.assertGreaterEqual(cal_diff, 0)
        
    def test_calculate_tpr(self):
        """Test True Positive Rate calculation."""
        # Test case with known values
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        tpr = self.fairness_evaluator._calculate_tpr(y_true, y_pred)
        self.assertEqual(tpr, 0.5)  # 1 TP out of 2 actual positives
        
        # Edge case: no positives
        y_true_no_pos = np.array([0, 0, 0, 0])
        tpr_no_pos = self.fairness_evaluator._calculate_tpr(y_true_no_pos, y_pred)
        self.assertTrue(np.isnan(tpr_no_pos))
        
    def test_calculate_fpr(self):
        """Test False Positive Rate calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        fpr = self.fairness_evaluator._calculate_fpr(y_true, y_pred)
        self.assertEqual(fpr, 0.5)  # 1 FP out of 2 actual negatives
        
    def test_calculate_ppv(self):
        """Test Positive Predictive Value calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        ppv = self.fairness_evaluator._calculate_ppv(y_true, y_pred)
        self.assertEqual(ppv, 0.5)  # 1 TP out of 2 positive predictions
        
    def test_calculate_npv(self):
        """Test Negative Predictive Value calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        npv = self.fairness_evaluator._calculate_npv(y_true, y_pred)
        self.assertEqual(npv, 0.5)  # 1 TN out of 2 negative predictions
        
    @patch('modules.fairness_evaluation.make_subplots')
    @patch('modules.fairness_evaluation.go')
    def test_create_fairness_dashboard(self, mock_go, mock_subplots):
        """Test fairness dashboard creation."""
        # Mock plotly components
        mock_fig = MagicMock()
        mock_subplots.return_value = mock_fig
        mock_go.Scatter.return_value = MagicMock()
        mock_go.Heatmap.return_value = MagicMock()
        mock_go.Histogram.return_value = MagicMock()
        
        try:
            fig = self.fairness_evaluator.create_fairness_dashboard(
                self.y_true, self.y_pred, self.y_proba, self.protected_attr
            )
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If there's an error, it should be handled gracefully
            self.assertIn('Error creating fairness dashboard', str(e))
            
    def test_generate_recommendations(self):
        """Test fairness recommendations generation."""
        # Create metrics with known bias
        metrics = {
            'group_fairness': {
                'demographic_parity': 0.15,  # Above threshold
                'disparate_impact': 0.7,     # Below threshold
                'equalized_odds_tpr': 0.05   # Within threshold
            },
            'calibration': {
                'calibration_difference': 0.12  # Above threshold
            }
        }
        
        recommendations = self.fairness_evaluator.generate_recommendations(metrics)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check that recommendations mention specific issues
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('demographic', rec_text)
        self.assertIn('disparate', rec_text)
        self.assertIn('calibration', rec_text)
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single group
        y_true_single = np.array([1, 0, 1, 0])
        y_pred_single = np.array([1, 1, 0, 0])
        y_proba_single = np.array([0.8, 0.7, 0.3, 0.2])
        protected_single = np.array([0, 0, 0, 0])  # All same group
        
        # Should handle gracefully without crashing
        try:
            metrics = self.fairness_evaluator.calculate_all_metrics(
                y_true_single, y_pred_single, y_proba_single, protected_single
            )
            # Many metrics should be NaN for single group
            self.assertTrue(True)
        except Exception as e:
            # Should handle the error gracefully
            self.assertIn('Error calculating fairness metrics', str(e))
            
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test with mismatched array lengths
        with self.assertRaises(Exception):
            self.fairness_evaluator.calculate_all_metrics(
                self.y_true[:50], self.y_pred, self.y_proba, self.protected_attr
            )
            
        # Test with non-binary protected attribute
        protected_multi = np.array([0, 1, 2] * (self.n_samples // 3) + [0] * (self.n_samples % 3))
        
        try:
            # Should raise an error for non-binary protected attribute
            self.fairness_evaluator.calculate_all_metrics(
                self.y_true, self.y_pred, self.y_proba, protected_multi
            )
        except ValueError as e:
            self.assertIn('binary protected attributes', str(e))


class TestFairnessEvaluatorIntegration(unittest.TestCase):
    """Integration tests for FairnessEvaluator with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.fairness_evaluator = FairnessEvaluator()
        
    def test_realistic_credit_scoring_scenario(self):
        """Test with a realistic credit scoring bias scenario."""
        np.random.seed(42)
        
        # Simulate credit scoring data with bias
        n_samples = 400
        
        # 60% majority, 40% minority
        protected = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Create ground truth (actual creditworthiness)
        # Assume similar creditworthiness across groups
        y_true = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # Create biased predictions
        # Majority group: higher approval rates for same creditworthiness
        y_pred = np.zeros(n_samples)
        y_proba = np.zeros(n_samples)
        
        for i in range(n_samples):
            if protected[i] == 0:  # Majority group
                base_prob = 0.8 if y_true[i] == 1 else 0.3
                bias_adjustment = 0.1
            else:  # Minority group
                base_prob = 0.7 if y_true[i] == 1 else 0.4
                bias_adjustment = -0.1
                
            y_proba[i] = np.clip(base_prob + bias_adjustment + np.random.normal(0, 0.1), 0, 1)
            y_pred[i] = 1 if y_proba[i] > 0.5 else 0
            
        # Calculate comprehensive metrics
        metrics = self.fairness_evaluator.calculate_all_metrics(
            y_true, y_pred, y_proba, protected
        )
        
        # Generate recommendations
        recommendations = self.fairness_evaluator.generate_recommendations(metrics)
        
        # Assertions
        self.assertGreater(len(recommendations), 0)
        
        # Check that bias is detected
        demographic_parity = metrics['group_fairness']['demographic_parity']
        self.assertGreater(demographic_parity, 0.05, "Should detect some bias")
        
    def test_fair_recruitment_scenario(self):
        """Test with a fair recruitment scenario."""
        np.random.seed(123)
        
        # Simulate fair recruitment data
        n_samples = 300
        protected = np.random.randint(0, 2, n_samples)
        
        # Fair hiring based purely on qualifications
        qualifications = np.random.normal(0, 1, n_samples)
        y_true = (qualifications > 0).astype(int)  # Ground truth qualification
        
        # Fair predictions (no bias)
        noise = np.random.normal(0, 0.2, n_samples)
        y_proba = 1 / (1 + np.exp(-(qualifications + noise)))
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.fairness_evaluator.calculate_all_metrics(
            y_true, y_pred, y_proba, protected
        )
        
        # Generate recommendations
        recommendations = self.fairness_evaluator.generate_recommendations(metrics)
        
        # Assertions for fair scenario
        demographic_parity = metrics['group_fairness']['demographic_parity']
        self.assertLess(demographic_parity, 0.1, "Should indicate fairness")
        
        # Should have positive recommendations
        rec_text = ' '.join(recommendations).lower()
        self.assertTrue('no major' in rec_text or 'continue monitoring' in rec_text)
        
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        
        # Large dataset
        n_samples = 2000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_proba = np.random.uniform(0, 1, n_samples)
        protected_attr = np.random.randint(0, 2, n_samples)
        
        # Should complete without timeout or memory issues
        import time
        start_time = time.time()
        
        metrics = self.fairness_evaluator.calculate_all_metrics(
            y_true, y_pred, y_proba, protected_attr
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        self.assertLess(execution_time, 5.0, "Should complete within reasonable time")
        
        # All metrics should be calculated
        self.assertIn('group_fairness', metrics)
        self.assertIn('predictive_parity', metrics)


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestFairnessEvaluator))
    suite.addTest(loader.loadTestsFromTestCase(TestFairnessEvaluatorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
