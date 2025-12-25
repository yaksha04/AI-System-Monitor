"""
Anomaly Detection Module
Uses ML to detect system anomalies
"""

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """ML-based anomaly detection for system metrics"""
    
    def __init__(self, config: Dict):
        """
        Initialize the anomaly detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config['ml']
        self.features = self.ml_config['features']
        self.model_file = config['storage']['model_file']
        self.scaler_file = config['storage']['scaler_file']
        
        # Ensure model directory exists
        Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_time = None
        
        # Load existing model if available
        self._load_model()
    
    def _create_model(self):
        """Create a new ML model based on configuration"""
        model_type = self.ml_config['model_type']
        contamination = self.ml_config['contamination']
        
        if model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
            logger.info("Created Isolation Forest model")
        elif model_type == 'one_class_svm':
            self.model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'
            )
            logger.info("Created One-Class SVM model")
        else:
            logger.error(f"Unknown model type: {model_type}. Using Isolation Forest.")
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
    
    def train(self, metrics_df: pd.DataFrame) -> bool:
        """
        Train the anomaly detection model
        
        Args:
            metrics_df: DataFrame with historical metrics
            
        Returns:
            True if training successful
        """
        try:
            if len(metrics_df) < 10:
                logger.warning(f"Not enough data to train (need at least 10 samples, have {len(metrics_df)})")
                return False
            
            # Check if all required features are present
            missing_features = [f for f in self.features if f not in metrics_df.columns]
            if missing_features:
                logger.error(f"Missing features for training: {missing_features}")
                return False
            
            # Prepare training data
            X = metrics_df[self.features].copy()
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Training model with {len(X)} samples and {len(self.features)} features")
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train model
            if self.model is None:
                self._create_model()
            
            self.model.fit(X_scaled)
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            
            # Save model
            self._save_model()
            
            logger.info(f"Model training completed successfully at {self.last_train_time}")
            
            # Log training statistics
            predictions = self.model.predict(X_scaled)
            anomaly_count = np.sum(predictions == -1)
            anomaly_percentage = (anomaly_count / len(predictions)) * 100
            logger.info(f"Training data analysis: {anomaly_count} anomalies detected ({anomaly_percentage:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return False
    
    def detect(self, metrics: Dict) -> Tuple[bool, float, Dict]:
        """
        Detect if current metrics are anomalous
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            Tuple of (is_anomaly, anomaly_score, details)
        """
        if not self.is_trained:
            logger.warning("Model not trained yet, cannot detect anomalies")
            return False, 0.0, {'status': 'model_not_trained'}
        
        try:
            # Extract features
            feature_values = []
            missing_features = []
            
            for feature in self.features:
                if feature in metrics:
                    value = metrics[feature]
                    # Handle NaN and inf
                    if pd.isna(value) or np.isinf(value):
                        value = 0
                    feature_values.append(value)
                else:
                    feature_values.append(0)
                    missing_features.append(feature)
            
            if missing_features:
                logger.debug(f"Missing features (using 0): {missing_features}")
            
            # Create feature array
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            is_anomaly = prediction == -1
            
            # Get anomaly score (lower is more anomalous)
            if hasattr(self.model, 'decision_function'):
                anomaly_score = self.model.decision_function(X_scaled)[0]
            elif hasattr(self.model, 'score_samples'):
                anomaly_score = self.model.score_samples(X_scaled)[0]
            else:
                anomaly_score = 0.0
            
            details = {
                'status': 'success',
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'timestamp': datetime.now().isoformat(),
                'features_used': len(self.features),
                'missing_features': missing_features
            }
            
            if is_anomaly:
                # Identify which features contributed most to anomaly
                details['anomalous_features'] = self._identify_anomalous_features(metrics)
            
            return is_anomaly, anomaly_score, details
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}", exc_info=True)
            return False, 0.0, {'status': 'error', 'error': str(e)}
    
    def _identify_anomalous_features(self, metrics: Dict) -> List[Dict]:
        """
        Identify which features are most anomalous
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            List of anomalous features with details
        """
        anomalous = []
        
        try:
            for feature in self.features:
                if feature not in metrics:
                    continue
                
                value = metrics[feature]
                
                # Simple heuristic: check if value is in extreme percentiles
                # This is a simplified approach
                if 'cpu' in feature.lower() and value > 80:
                    anomalous.append({
                        'feature': feature,
                        'value': value,
                        'reason': 'High CPU usage'
                    })
                elif 'memory' in feature.lower() and value > 85:
                    anomalous.append({
                        'feature': feature,
                        'value': value,
                        'reason': 'High memory usage'
                    })
                elif 'disk' in feature.lower() and 'percent' in feature.lower() and value > 90:
                    anomalous.append({
                        'feature': feature,
                        'value': value,
                        'reason': 'High disk usage'
                    })
                    
        except Exception as e:
            logger.error(f"Error identifying anomalous features: {e}")
        
        return anomalous
    
    def should_retrain(self) -> bool:
        """
        Check if model should be retrained
        
        Returns:
            True if retraining is needed
        """
        if not self.is_trained or self.last_train_time is None:
            return True
        
        retrain_interval = timedelta(hours=self.ml_config['retrain_interval_hours'])
        time_since_training = datetime.now() - self.last_train_time
        
        return time_since_training >= retrain_interval
    
    def _save_model(self):
        """Save model and scaler to disk"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Model saved to {self.model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load model and scaler from disk"""
        try:
            if Path(self.model_file).exists() and Path(self.scaler_file).exists():
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                self.last_train_time = datetime.fromtimestamp(Path(self.model_file).stat().st_mtime)
                
                logger.info(f"Model loaded from {self.model_file} (trained at {self.last_train_time})")
                
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False
    
    def get_feature_importance(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance scores (simplified version)
        
        Args:
            metrics_df: DataFrame with metrics
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained or len(metrics_df) < 10:
            return {}
        
        try:
            importance = {}
            
            for feature in self.features:
                if feature in metrics_df.columns:
                    # Use variance as a simple importance measure
                    variance = metrics_df[feature].var()
                    importance[feature] = float(variance)
            
            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
