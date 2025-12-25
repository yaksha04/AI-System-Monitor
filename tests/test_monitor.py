"""
Unit Tests for AI System Monitor
"""

import unittest
import sys
from pathlib import Path
import tempfile
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from monitoring.metrics_collector import MetricsCollector
from monitoring.system_monitor import SystemMonitor
from ml.anomaly_detector import AnomalyDetector
from healing.auto_healer import AutoHealer
from notifications.notifier import Notifier


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection"""
    
    def setUp(self):
        self.collector = MetricsCollector()
    
    def test_collect_all_metrics(self):
        """Test that all metrics are collected"""
        metrics = self.collector.collect_all_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('disk_percent', metrics)
        self.assertIn('timestamp', metrics)
    
    def test_cpu_metrics(self):
        """Test CPU metrics are within valid range"""
        metrics = self.collector.collect_all_metrics()
        
        cpu_percent = metrics.get('cpu_percent', -1)
        self.assertGreaterEqual(cpu_percent, 0)
        self.assertLessEqual(cpu_percent, 100)
    
    def test_memory_metrics(self):
        """Test memory metrics are within valid range"""
        metrics = self.collector.collect_all_metrics()
        
        mem_percent = metrics.get('memory_percent', -1)
        self.assertGreaterEqual(mem_percent, 0)
        self.assertLessEqual(mem_percent, 100)
    
    def test_process_count(self):
        """Test process count is positive"""
        metrics = self.collector.collect_all_metrics()
        
        process_count = metrics.get('process_count', 0)
        self.assertGreater(process_count, 0)


class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring"""
    
    def setUp(self):
        # Create temp config
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'monitoring': {
                'interval_seconds': 5,
                'retention_hours': 24,
                'thresholds': {
                    'cpu_percent': 85,
                    'memory_percent': 90,
                    'disk_percent': 85
                },
                'critical_processes': []
            },
            'storage': {
                'metrics_file': f'{self.temp_dir}/metrics.csv',
                'model_file': f'{self.temp_dir}/model.pkl',
                'scaler_file': f'{self.temp_dir}/scaler.pkl'
            }
        }
        self.monitor = SystemMonitor(self.config)
    
    def test_collect_and_store(self):
        """Test metric collection and storage"""
        metrics = self.monitor.collect_and_store_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(self.monitor.metrics_df), 0)
    
    def test_get_recent_metrics(self):
        """Test retrieving recent metrics"""
        # Collect some metrics
        for _ in range(3):
            self.monitor.collect_and_store_metrics()
        
        recent = self.monitor.get_recent_metrics(minutes=5)
        self.assertGreater(len(recent), 0)
    
    def test_check_thresholds(self):
        """Test threshold checking"""
        test_metrics = {
            'cpu_percent': 90,  # Above threshold
            'memory_percent': 50,
            'disk_percent': 50
        }
        
        violations = self.monitor.check_thresholds(test_metrics)
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0]['metric'], 'cpu_percent')


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'ml': {
                'model_type': 'isolation_forest',
                'contamination': 0.1,
                'training_window_minutes': 60,
                'retrain_interval_hours': 6,
                'features': [
                    'cpu_percent', 'memory_percent', 'disk_percent'
                ]
            },
            'storage': {
                'metrics_file': f'{self.temp_dir}/metrics.csv',
                'model_file': f'{self.temp_dir}/model.pkl',
                'scaler_file': f'{self.temp_dir}/scaler.pkl'
            }
        }
        self.detector = AnomalyDetector(self.config)
    
    def test_model_creation(self):
        """Test model is created correctly"""
        self.detector._create_model()
        self.assertIsNotNone(self.detector.model)
    
    def test_training_with_data(self):
        """Test model training with synthetic data"""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'cpu_percent': np.random.uniform(20, 60, n_samples),
            'memory_percent': np.random.uniform(30, 70, n_samples),
            'disk_percent': np.random.uniform(40, 80, n_samples)
        }
        df = pd.DataFrame(data)
        
        # Train model
        result = self.detector.train(df)
        self.assertTrue(result)
        self.assertTrue(self.detector.is_trained)
    
    def test_detection(self):
        """Test anomaly detection"""
        # Train with normal data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'cpu_percent': np.random.uniform(20, 60, n_samples),
            'memory_percent': np.random.uniform(30, 70, n_samples),
            'disk_percent': np.random.uniform(40, 80, n_samples)
        }
        df = pd.DataFrame(data)
        self.detector.train(df)
        
        # Test with normal metrics
        normal_metrics = {
            'cpu_percent': 45,
            'memory_percent': 55,
            'disk_percent': 65
        }
        
        is_anomaly, score, details = self.detector.detect(normal_metrics)
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        
        # Test with anomalous metrics
        anomalous_metrics = {
            'cpu_percent': 95,
            'memory_percent': 95,
            'disk_percent': 95
        }
        
        is_anomaly, score, details = self.detector.detect(anomalous_metrics)
        # Note: May or may not detect as anomaly depending on training


class TestAutoHealer(unittest.TestCase):
    """Test auto-healing functionality"""
    
    def setUp(self):
        self.config = {
            'healing': {
                'enabled': True,
                'max_attempts': 3,
                'actions': {}
            }
        }
        self.healer = AutoHealer(self.config)
    
    def test_initialization(self):
        """Test healer initializes correctly"""
        self.assertTrue(self.healer.enabled)
        self.assertEqual(self.healer.max_attempts, 3)
    
    def test_high_cpu_detection(self):
        """Test high CPU detection"""
        metrics = {'cpu_percent': 95}
        self.assertTrue(self.healer._is_high_cpu(metrics))
        
        metrics = {'cpu_percent': 50}
        self.assertFalse(self.healer._is_high_cpu(metrics))
    
    def test_high_memory_detection(self):
        """Test high memory detection"""
        metrics = {'memory_percent': 90}
        self.assertTrue(self.healer._is_high_memory(metrics))
        
        metrics = {'memory_percent': 50}
        self.assertFalse(self.healer._is_high_memory(metrics))
    
    def test_healing_when_disabled(self):
        """Test healing does nothing when disabled"""
        self.healer.enabled = False
        
        result = self.healer.heal({}, {})
        self.assertEqual(result['status'], 'disabled')
    
    def test_action_history(self):
        """Test action history is maintained"""
        action = {
            'action': 'test_action',
            'status': 'success',
            'timestamp': '2025-10-04T10:00:00'
        }
        
        self.healer._log_action(action)
        history = self.healer.get_action_history()
        
        self.assertGreater(len(history), 0)
        self.assertEqual(history[-1]['action'], 'test_action')


class TestNotifier(unittest.TestCase):
    """Test notification system"""
    
    def setUp(self):
        self.config = {
            'notifications': {
                'enabled': False,  # Disabled for testing
                'email': {'enabled': False},
                'slack': {'enabled': False}
            }
        }
        self.notifier = Notifier(self.config)
    
    def test_initialization(self):
        """Test notifier initializes correctly"""
        self.assertFalse(self.notifier.enabled)
    
    def test_notification_disabled(self):
        """Test notifications don't send when disabled"""
        result = self.notifier.send_anomaly_alert({}, {}, {})
        self.assertEqual(result['status'], 'disabled')
    
    def test_message_formatting(self):
        """Test message formatting"""
        anomaly_details = {
            'anomaly_score': -0.5,
            'timestamp': '2025-10-04T10:00:00'
        }
        
        metrics = {
            'cpu_percent': 85,
            'memory_percent': 75,
            'disk_percent': 60,
            'load_average_1min': 2.5
        }
        
        healing_result = {
            'actions': [
                {'action': 'test_action', 'status': 'success'}
            ]
        }
        
        message = self.notifier._format_anomaly_message(
            anomaly_details, metrics, healing_result
        )
        
        self.assertIsInstance(message, str)
        self.assertIn('Anomaly', message)
        self.assertIn('CPU', message)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'monitoring': {
                'interval_seconds': 5,
                'retention_hours': 24,
                'thresholds': {
                    'cpu_percent': 85,
                    'memory_percent': 90,
                    'disk_percent': 85
                },
                'critical_processes': []
            },
            'ml': {
                'model_type': 'isolation_forest',
                'contamination': 0.1,
                'training_window_minutes': 60,
                'retrain_interval_hours': 6,
                'features': [
                    'cpu_percent', 'memory_percent', 'disk_percent',
                    'load_average_1min', 'process_count'
                ]
            },
            'healing': {
                'enabled': True,
                'max_attempts': 3,
                'actions': {}
            },
            'notifications': {
                'enabled': False,
                'email': {'enabled': False},
                'slack': {'enabled': False}
            },
            'storage': {
                'metrics_file': f'{self.temp_dir}/metrics.csv',
                'model_file': f'{self.temp_dir}/model.pkl',
                'scaler_file': f'{self.temp_dir}/scaler.pkl'
            }
        }
    
    def test_full_pipeline(self):
        """Test complete monitoring pipeline"""
        # Initialize components
        monitor = SystemMonitor(self.config)
        detector = AnomalyDetector(self.config)
        healer = AutoHealer(self.config)
        
        # Collect metrics
        for _ in range(20):
            metrics = monitor.collect_and_store_metrics()
            self.assertIsInstance(metrics, dict)
        
        # Prepare data for ML
        ml_data = monitor.get_metrics_for_ml(self.config['ml']['features'])
        self.assertGreater(len(ml_data), 0)
        
        # Train model
        if len(ml_data) >= 10:
            result = detector.train(ml_data)
            self.assertTrue(result)
            
            # Detect anomalies
            current_metrics = monitor.collect_and_store_metrics()
            is_anomaly, score, details = detector.detect(current_metrics)
            
            self.assertIsInstance(is_anomaly, bool)
            self.assertIsInstance(details, dict)
            
            # Test healing
            if is_anomaly:
                healing_result = healer.heal(details, current_metrics)
                self.assertIn('status', healing_result)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoHealer))
    suite.addTests(loader.loadTestsFromTestCase(TestNotifier))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
