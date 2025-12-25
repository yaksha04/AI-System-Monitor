"""
Main Application Entry Point
AI-Powered Intelligent Linux System Monitor & Auto-Healer
"""

import yaml
import logging
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
import colorlog

# Import modules
from monitoring.system_monitor import SystemMonitor
from ml.anomaly_detector import AnomalyDetector
from healing.auto_healer import AutoHealer
from notifications.notifier import Notifier


class SystemMonitorApp:
    """Main application class"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the application
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AI System Monitor...")
        
        # Initialize components
        self.monitor = SystemMonitor(self.config)
        self.detector = AnomalyDetector(self.config)
        self.healer = AutoHealer(self.config)
        self.notifier = Notifier(self.config)
        
        # Runtime state
        self.running = False
        self.iteration_count = 0
        self.anomaly_count = 0
        self.healing_action_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("AI System Monitor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {config_path}")
            print("Creating default configuration...")
            self._create_default_config(config_path)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _create_default_config(self, config_path: str):
        """Create default configuration file"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
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
                    'cpu_percent', 'memory_percent', 'memory_available_mb',
                    'disk_io_read_mb', 'disk_io_write_mb',
                    'network_sent_mb', 'network_recv_mb',
                    'process_count', 'load_average_1min', 'load_average_5min'
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
            'dashboard': {
                'port': 8501,
                'host': '0.0.0.0',
                'refresh_interval_seconds': 5
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': 'data/logs/system_monitor.log',
                'max_file_size_mb': 100,
                'backup_count': 5
            },
            'storage': {
                'metrics_file': 'data/metrics/metrics.csv',
                'model_file': 'data/models/anomaly_model.pkl',
                'scaler_file': 'data/models/scaler.pkl'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        
        # Create logs directory
        if log_config.get('log_to_file'):
            log_file = log_config['log_file']
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup color logging for console
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        
        # Setup file handler
        handlers = [console_handler]
        
        if log_config.get('log_to_file'):
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config['log_file'],
                maxBytes=log_config.get('max_file_size_mb', 100) * 1024 * 1024,
                backupCount=log_config.get('backup_count', 5)
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            handlers=handlers
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main application loop"""
        self.running = True
        self.logger.info("Starting monitoring loop...")
        
        interval = self.config['monitoring']['interval_seconds']
        training_window_minutes = self.config['ml']['training_window_minutes']
        
        # Initial training check
        training_metrics_needed = (training_window_minutes * 60) // interval
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Collect metrics
                metrics = self.monitor.collect_and_store_metrics()
                
                if not metrics:
                    self.logger.error("Failed to collect metrics, skipping iteration")
                    time.sleep(interval)
                    continue
                
                self.iteration_count += 1
                
                # Check if we have enough data for training
                if not self.detector.is_trained:
                    total_metrics = len(self.monitor.metrics_df)
                    if total_metrics >= training_metrics_needed:
                        self.logger.info(f"Collected {total_metrics} metrics, starting initial training...")
                        training_data = self.monitor.get_recent_metrics(minutes=training_window_minutes)
                        ml_data = self.monitor.get_metrics_for_ml(self.config['ml']['features'])
                        
                        if self.detector.train(ml_data):
                            self.logger.info("Initial model training completed successfully")
                        else:
                            self.logger.warning("Initial training failed, will retry")
                    else:
                        self.logger.info(f"Collecting initial data: {total_metrics}/{training_metrics_needed}")
                        time.sleep(interval)
                        continue
                
                # Check if retraining is needed
                if self.detector.should_retrain():
                    self.logger.info("Retraining model with recent data...")
                    training_data = self.monitor.get_recent_metrics(minutes=training_window_minutes)
                    ml_data = self.monitor.get_metrics_for_ml(self.config['ml']['features'])
                    
                    if self.detector.train(ml_data):
                        self.logger.info("✅ Model retraining completed")
                
                # Detect anomalies
                is_anomaly, score, details = self.detector.detect(metrics)
                
                if is_anomaly:
                    self.anomaly_count += 1
                    self.logger.warning(f"🚨 ANOMALY DETECTED! Score: {score:.4f}")
                    
                    # Perform auto-healing
                    healing_result = self.healer.heal(details, metrics)
                    
                    if healing_result['action_count'] > 0:
                        self.healing_action_count += healing_result['action_count']
                        self.logger.info(f"🔧 Auto-healing performed {healing_result['action_count']} action(s)")
                    
                    # Send notifications
                    notif_result = self.notifier.send_anomaly_alert(details, metrics, healing_result)
                    if notif_result['status'] == 'sent':
                        self.logger.info(f"📧 Notifications sent via {notif_result['channels']} channel(s)")
                
                # Check thresholds
                threshold_violations = self.monitor.check_thresholds(metrics)
                if threshold_violations:
                    self.logger.warning(f"⚠️ {len(threshold_violations)} threshold violation(s) detected")
                    self.notifier.send_threshold_alert(threshold_violations)
                
                # Check critical processes
                service_issues = self.monitor.check_critical_processes()
                if service_issues:
                    self.logger.critical(f"🔴 {len(service_issues)} critical service(s) down!")
                    self.notifier.send_service_down_alert(service_issues)
                
                # Log status every 20 iterations
                if self.iteration_count % 20 == 0:
                    self._log_status(metrics)
                
                # Sleep for remaining interval time
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self._shutdown()
    
    def _log_status(self, metrics: Dict):
        """Log current status summary"""
        self.logger.info("=" * 60)
        self.logger.info("STATUS SUMMARY")
        self.logger.info(f"  Runtime: {self.iteration_count} iterations")
        self.logger.info(f"  Anomalies detected: {self.anomaly_count}")
        self.logger.info(f"  Healing actions: {self.healing_action_count}")
        self.logger.info(f"  CPU: {metrics.get('cpu_percent', 0):.1f}% | "
                        f"Memory: {metrics.get('memory_percent', 0):.1f}% | "
                        f"Disk: {metrics.get('disk_percent', 0):.1f}%")
        stats = self.monitor.get_statistics()
        if stats:
            self.logger.info(f"  Total metrics collected: {stats.get('total_metrics', 0)}")
        self.logger.info("=" * 60)
    
    def _shutdown(self):
        """Cleanup and shutdown"""
        self.logger.info("Shutting down AI System Monitor...")
        self.logger.info(f"Final statistics:")
        self.logger.info(f"  - Total iterations: {self.iteration_count}")
        self.logger.info(f"  - Anomalies detected: {self.anomaly_count}")
        self.logger.info(f"  - Healing actions performed: {self.healing_action_count}")
        self.logger.info("Goodbye! 👋")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   AI-Powered Intelligent Linux System Monitor            ║
    ║                  & Auto-Healer                            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app = SystemMonitorApp()
    app.run()


if __name__ == '__main__':
    main()