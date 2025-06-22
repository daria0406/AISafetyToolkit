import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.bias_detection import BiasDetector

class TestBiasDetector(unittest.TestCase):
    """Test cases for the BiasDetector class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.bias_detector = BiasDetector()
        
        # Create sample data for testing
        np.random.seed(42)
        self.n_samples = 100
        
        # Balanced groups
        self.y_pred_balanced = np.random.randint(0, 2, self.n_samples)
        self.protected_balanced = np.random.randint(0, 2, self.n_samples)
        self.y_true_balanced = np.random.randint(0, 2, self.n_samples)
        
        # Biased data (group 1 has lower positive rate)
        self.y_pred_biased = np.array([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)  # 60% vs 20%
        self.protected_biased = np.array([0] * 50 + [1] * 50)  # 50-50 split
        self.y_true_biased = np.array([1] * 25 + [0] * 25 + [1] * 15 + [0] * 35)
        
    def test_init(self):
        """Test BiasDetector initialization."""
        detector = BiasDetector()
        self.assertEqual(detector.bias_threshold, 0.8)
        
    def test_calculate_disparate_impact_balanced(self):
        """Test disparate impact calculation with balanced data."""
        di = self.bias_detector.calculate_disparate_impact(
            self.y_pred_balanced, self.protected_balanced
        )
        self.assertIsInstance(di, (int, float))
        self.assertGreater(di, 0)
        
    def test_calculate_disparate_impact_biased(self):
        """Test disparate impact calculation with biased data."""
        di = self.bias_detector.calculate_disparate_impact(
            self.y_pred_biased, self.protected_biased
        )
        
        # Group 0: 30/50 = 0.6, Group 1: 10/50 = 0.2
        # DI = 0.2/0.6 = 0.333
        expected_di = 0.2 / 0.6
        self.assertAlmostEqual(di, expected_di, places=2)
        
    def test_calculate_disparate_impact_edge_cases(self):
        """Test disparate impact calculation with edge cases."""
        # All zeros
        y_pred_zeros = np.zeros(10)
        protected = np.array([0] * 5 + [1] * 5)
        di = self.bias_detector.calculate_disparate_impact(y_pred_zeros, protected)
        self.assertEqual(di, 1.0)  # 0/0 case should return 1.0
        
        # Single group
        y_pred = np.array([1, 0, 1, 0])
        protected_single = np.array([0, 0, 0, 0])
        di = self.bias_detector.calculate_disparate_impact(y_pred, protected_single)
        self.assertTrue(np.isnan(di))
        
    def test_calculate_statistical_parity(self):
        """Test statistical parity calculation."""
        sp = self.bias_detector.calculate_statistical_parity(
            self.y_pred_biased, self.protected_biased
        )
        
        # Group 0: 30/50 = 0.6, Group 1: 10/50 = 0.2
        # SP = 0.2 - 0.6 = -0.4
        expected_sp = 0.2 - 0.6
        self.assertAlmostEqual(sp, expected_sp, places=2)
        
    def test_calculate_equalized_odds(self):
        """Test equalized odds calculation."""
        eo = self.bias_detector.calculate_equalized_odds(
            self.y_true_biased, self.y_pred_biased, self.protected_biased
        )
        self.assertIsInstance(eo, (int, float))
        self.assertGreaterEqual(eo, 0)
        
    def test_calculate_tpr(self):
        """Test True Positive Rate calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        tpr = self.bias_detector._calculate_tpr(y_true, y_pred)
        self.assertEqual(tpr, 0.5)  # 1 TP out of 2 actual positives
        
        # Edge case: no positives
        y_true_no_pos = np.array([0, 0, 0, 0])
        tpr_no_pos = self.bias_detector._calculate_tpr(y_true_no_pos, y_pred)
        self.assertTrue(np.isnan(tpr_no_pos))
        
    def test_calculate_demographic_parity(self):
        """Test demographic parity calculation."""
        dp = self.bias_detector.calculate_demographic_parity(
            self.y_pred_biased, self.protected_biased
        )
        
        self.assertIn('group_rates', dp)
        self.assertIn('max_difference', dp)
        self.assertIn('is_fair', dp)
        
        # Check calculated values
        self.assertAlmostEqual(dp['group_rates']['group_0'], 0.6, places=2)
        self.assertAlmostEqual(dp['group_rates']['group_1'], 0.2, places=2)
        self.assertAlmostEqual(dp['max_difference'], 0.4, places=2)
        self.assertFalse(dp['is_fair'])  # 0.4 > 0.1 threshold
        
    def test_generate_bias_report(self):
        """Test comprehensive bias report generation."""
        report = self.bias_detector.generate_bias_report(
            self.y_true_biased, self.y_pred_biased, self.protected_biased
        )
        
        # Check all required metrics are present
        required_metrics = ['disparate_impact', 'statistical_parity', 
                          'equalized_odds', 'demographic_parity']
        for metric in required_metrics:
            self.assertIn(metric, report)
        
        # Check interpretations are generated
        self.assertIn('interpretations', report)
        self.assertIsInstance(report['interpretations'], list)
        self.assertGreater(len(report['interpretations']), 0)
        
    def test_generate_interpretations(self):
        """Test interpretation generation."""
        # Create mock metrics with known bias
        metrics = {
            'disparate_impact': 0.5,  # Below threshold
            'statistical_parity': 0.15,  # Above threshold
            'equalized_odds': 0.05  # Within threshold
        }
        
        interpretations = self.bias_detector._generate_interpretations(metrics)
        
        self.assertIsInstance(interpretations, list)
        self.assertGreater(len(interpretations), 0)
        
        # Check specific interpretations
        interp_text = ' '.join(interpretations)
        self.assertIn('bias', interp_text.lower())
        
    @patch('modules.bias_detection.make_subplots')
    @patch('modules.bias_detection.go')
    def test_plot_bias_metrics(self, mock_go, mock_subplots):
        """Test bias metrics visualization."""
        # Mock plotly components
        mock_fig = MagicMock()
        mock_subplots.return_value = mock_fig
        mock_go.Bar.return_value = MagicMock()
        mock_go.Heatmap.return_value = MagicMock()
        mock_go.Pie.return_value = MagicMock()
        
        try:
            fig = self.bias_detector.plot_bias_metrics(
                self.y_pred_biased, self.protected_biased, self.y_true_biased
            )
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If there's an error, it should be handled gracefully
            self.assertIn('Error creating bias visualization', str(e))
            
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test with empty arrays
        with self.assertRaises(Exception):
            self.bias_detector.calculate_disparate_impact([], [])
            
        # Test with mismatched array lengths
        with self.assertRaises(Exception):
            self.bias_detector.calculate_disparate_impact([1, 0], [1])
            
        # Test with non-binary protected attribute (should handle gracefully)
        y_pred = np.array([1, 0, 1, 0])
        protected_multi = np.array([0, 1, 2, 0])  # 3 groups
        
        # Should not raise exception but handle appropriately
        try:
            di = self.bias_detector.calculate_disparate_impact(y_pred, protected_multi)
            # Should return some value or NaN
            self.assertTrue(np.isnan(di) or isinstance(di, (int, float)))
        except Exception:
            pass  # Some methods may not support multi-group
            
    def test_numpy_array_conversion(self):
        """Test that inputs are properly converted to numpy arrays."""
        # Test with lists
        y_pred_list = [1, 0, 1, 0]
        protected_list = [0, 1, 0, 1]
        
        di = self.bias_detector.calculate_disparate_impact(y_pred_list, protected_list)
        self.assertIsInstance(di, (int, float))
        
        # Test with pandas Series
        y_pred_series = pd.Series([1, 0, 1, 0])
        protected_series = pd.Series([0, 1, 0, 1])
        
        di = self.bias_detector.calculate_disparate_impact(y_pred_series, protected_series)
        self.assertIsInstance(di, (int, float))


class TestBiasDetectorIntegration(unittest.TestCase):
    """Integration tests for BiasDetector with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.bias_detector = BiasDetector()
        
    def test_realistic_hiring_scenario(self):
        """Test with a realistic hiring bias scenario."""
        np.random.seed(42)
        
        # Simulate hiring data with bias against minority group
        n_samples = 500
        
        # 70% majority, 30% minority
        protected = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Create biased hiring decisions
        # Majority group: 60% hire rate
        # Minority group: 30% hire rate
        y_pred = np.zeros(n_samples)
        majority_mask = protected == 0
        minority_mask = protected == 1
        
        y_pred[majority_mask] = np.random.choice([0, 1], np.sum(majority_mask), p=[0.4, 0.6])
        y_pred[minority_mask] = np.random.choice([0, 1], np.sum(minority_mask), p=[0.7, 0.3])
        
        # Generate corresponding ground truth
        y_true = np.random.randint(0, 2, n_samples)
        
        # Calculate metrics
        di = self.bias_detector.calculate_disparate_impact(y_pred, protected)
        sp = self.bias_detector.calculate_statistical_parity(y_pred, protected)
        eo = self.bias_detector.calculate_equalized_odds(y_true, y_pred, protected)
        
        # Assertions for biased scenario
        self.assertLess(di, 0.8, "Disparate impact should indicate bias")
        self.assertLess(sp, -0.1, "Statistical parity should indicate bias against minority")
        
        # Generate report
        report = self.bias_detector.generate_bias_report(y_true, y_pred, protected)
        self.assertIn('bias', ' '.join(report['interpretations']).lower())
        
    def test_fair_lending_scenario(self):
        """Test with a fair lending scenario."""
        np.random.seed(123)
        
        # Simulate fair lending data
        n_samples = 300
        protected = np.random.randint(0, 2, n_samples)
        
        # Fair decisions (similar approval rates)
        approval_rate = 0.7
        y_pred = np.random.choice([0, 1], n_samples, p=[1-approval_rate, approval_rate])
        y_true = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # Calculate metrics
        di = self.bias_detector.calculate_disparate_impact(y_pred, protected)
        sp = self.bias_detector.calculate_statistical_parity(y_pred, protected)
        
        # Assertions for fair scenario
        self.assertGreater(di, 0.8, "Disparate impact should indicate fairness")
        self.assertLess(abs(sp), 0.1, "Statistical parity should indicate fairness")


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestBiasDetector))
    suite.addTest(loader.loadTestsFromTestCase(TestBiasDetectorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
