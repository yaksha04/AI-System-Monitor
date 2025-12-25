"""
System Monitor Module
Coordinates metric collection and storage
"""

import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self, config: Dict):
        """
        Initialize the system monitor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.collector = MetricsCollector()
        self.metrics_file = config['storage']['metrics_file']
        self.retention_hours = config['monitoring']['retention_hours']
        
        # Ensure data directory exists
        Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metrics dataframe
        self.metrics_df = self._load_or_create_metrics_df()
        
    def _load_or_create_metrics_df(self) -> pd.DataFrame:
        """Load existing metrics or create new dataframe"""
        if os.path.exists(self.metrics_file):
            try:
                df = pd.read_csv(self.metrics_file)
                logger.info(f"Loaded {len(df)} existing metrics from {self.metrics_file}")
                return df
            except Exception as e:
                logger.warning(f"Error loading metrics file: {e}. Creating new file.")
        
        return pd.DataFrame()
    
    def collect_and_store_metrics(self) -> Dict:
        """
        Collect current metrics and store them
        
        Returns:
            Current metrics dictionary
        """
        try:
            # Collect metrics
            metrics = self.collector.collect_all_metrics()
            
            if not metrics:
                logger.error("Failed to collect metrics")
                return {}
            
            # Add to dataframe
            self.metrics_df = pd.concat([
                self.metrics_df,
                pd.DataFrame([metrics])
            ], ignore_index=True)
            
            # Clean old data
            self._clean_old_metrics()
            
            # Save to file
            self._save_metrics()
            
            logger.debug(f"Collected and stored metrics: CPU {metrics.get('cpu_percent', 0):.1f}%, "
                        f"Memory {metrics.get('memory_percent', 0):.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in collect_and_store_metrics: {e}")
            return {}
    
    def _clean_old_metrics(self):
        """Remove metrics older than retention period"""
        if len(self.metrics_df) == 0:
            return
        
        try:
            # Convert timestamp to datetime
            self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            # Filter out old metrics
            initial_count = len(self.metrics_df)
            self.metrics_df = self.metrics_df[self.metrics_df['timestamp'] >= cutoff_time]
            removed_count = initial_count - len(self.metrics_df)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old metrics (older than {self.retention_hours} hours)")
                
        except Exception as e:
            logger.error(f"Error cleaning old metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to CSV file"""
        try:
            self.metrics_df.to_csv(self.metrics_file, index=False)
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")
    
    def get_recent_metrics(self, minutes: int = 60) -> pd.DataFrame:
        """
        Get metrics from the last N minutes
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            DataFrame with recent metrics
        """
        if len(self.metrics_df) == 0:
            return pd.DataFrame()
        
        try:
            self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            return self.metrics_df[self.metrics_df['timestamp'] >= cutoff_time].copy()
        except Exception as e:
            logger.error(f"Error getting recent metrics: {e}")
            return pd.DataFrame()
    
    def get_metrics_for_ml(self, features: List[str]) -> pd.DataFrame:
        """
        Get metrics formatted for ML model
        
        Args:
            features: List of feature column names
            
        Returns:
            DataFrame with specified features
        """
        if len(self.metrics_df) == 0:
            return pd.DataFrame(columns=features)
        
        try:
            # Select only the specified features that exist
            available_features = [f for f in features if f in self.metrics_df.columns]
            
            if not available_features:
                logger.warning("No requested features available in metrics")
                return pd.DataFrame(columns=features)
            
            df = self.metrics_df[available_features].copy()
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Add missing features as zeros
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0
            
            return df[features]
            
        except Exception as e:
            logger.error(f"Error preparing metrics for ML: {e}")
            return pd.DataFrame(columns=features)
    
    def check_thresholds(self, metrics: Dict) -> List[Dict]:
        """
        Check if metrics exceed configured thresholds
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            List of threshold violations
        """
        violations = []
        thresholds = self.config['monitoring']['thresholds']
        
        checks = [
            ('cpu_percent', 'CPU Usage'),
            ('memory_percent', 'Memory Usage'),
            ('disk_percent', 'Disk Usage'),
        ]
        
        for metric_key, description in checks:
            if metric_key in metrics and metric_key in thresholds:
                current_value = metrics[metric_key]
                threshold_value = thresholds[metric_key]
                
                if current_value > threshold_value:
                    violations.append({
                        'metric': metric_key,
                        'description': description,
                        'current_value': current_value,
                        'threshold': threshold_value,
                        'severity': 'high' if current_value > threshold_value + 5 else 'medium'
                    })
        
        return violations
    
    def check_critical_processes(self) -> List[Dict]:
        """
        Check status of critical processes
        
        Returns:
            List of process status issues
        """
        issues = []
        critical_processes = self.config['monitoring'].get('critical_processes', [])
        
        if not critical_processes:
            return issues
        
        process_status = self.collector.check_critical_processes(critical_processes)
        
        for process_name, is_running in process_status.items():
            if not is_running:
                issues.append({
                    'process': process_name,
                    'status': 'not_running',
                    'severity': 'critical'
                })
        
        return issues
    
    def get_statistics(self) -> Dict:
        """
        Get statistical summary of metrics
        
        Returns:
            Dictionary with statistics
        """
        if len(self.metrics_df) == 0:
            return {}
        
        try:
            recent_df = self.get_recent_metrics(minutes=60)
            
            if len(recent_df) == 0:
                return {}
            
            stats = {
                'cpu': {
                    'current': recent_df['cpu_percent'].iloc[-1] if len(recent_df) > 0 else 0,
                    'average': recent_df['cpu_percent'].mean(),
                    'max': recent_df['cpu_percent'].max(),
                    'min': recent_df['cpu_percent'].min(),
                },
                'memory': {
                    'current': recent_df['memory_percent'].iloc[-1] if len(recent_df) > 0 else 0,
                    'average': recent_df['memory_percent'].mean(),
                    'max': recent_df['memory_percent'].max(),
                    'min': recent_df['memory_percent'].min(),
                },
                'disk': {
                    'current': recent_df['disk_percent'].iloc[-1] if len(recent_df) > 0 else 0,
                    'average': recent_df['disk_percent'].mean(),
                    'max': recent_df['disk_percent'].max(),
                    'min': recent_df['disk_percent'].min(),
                },
                'total_metrics': len(self.metrics_df),
                'recent_metrics': len(recent_df),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}