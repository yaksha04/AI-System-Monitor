"""
Auto-Healing Module
Automatically fixes detected system issues
"""

import subprocess
import psutil
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoHealer:
    """Automatically heals system issues"""
    
    def __init__(self, config: Dict):
        """
        Initialize the auto-healer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.healing_config = config['healing']
        self.enabled = self.healing_config['enabled']
        self.max_attempts = self.healing_config['max_attempts']
        self.action_history = []
        
    def heal(self, anomaly_details: Dict, metrics: Dict) -> Dict:
        """
        Perform healing actions based on anomaly
        
        Args:
            anomaly_details: Details from anomaly detection
            metrics: Current system metrics
            
        Returns:
            Dictionary with healing results
        """
        if not self.enabled:
            logger.info("Auto-healing is disabled")
            return {'status': 'disabled', 'actions': []}
        
        logger.info(f"Starting auto-healing process for anomaly: {anomaly_details}")
        
        actions_taken = []
        
        # Determine issue type and take appropriate action
        if self._is_high_cpu(metrics):
            actions_taken.extend(self._handle_high_cpu(metrics))
        
        if self._is_high_memory(metrics):
            actions_taken.extend(self._handle_high_memory(metrics))
        
        if self._is_disk_full(metrics):
            actions_taken.extend(self._handle_disk_full(metrics))
        
        # Check for specific anomalous features
        if 'anomalous_features' in anomaly_details:
            for feature_info in anomaly_details['anomalous_features']:
                feature = feature_info['feature']
                
                if 'load_average' in feature:
                    actions_taken.extend(self._handle_high_load())
        
        # Log actions
        for action in actions_taken:
            self._log_action(action)
        
        result = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'actions': actions_taken,
            'action_count': len(actions_taken)
        }
        
        logger.info(f"Auto-healing completed: {len(actions_taken)} actions taken")
        
        return result
    
    def _is_high_cpu(self, metrics: Dict) -> bool:
        """Check if CPU usage is critically high"""
        return metrics.get('cpu_percent', 0) > 90
    
    def _is_high_memory(self, metrics: Dict) -> bool:
        """Check if memory usage is critically high"""
        return metrics.get('memory_percent', 0) > 85
    
    def _is_disk_full(self, metrics: Dict) -> bool:
        """Check if disk is critically full"""
        return metrics.get('disk_percent', 0) > 90
    
    def _handle_high_cpu(self, metrics: Dict) -> List[Dict]:
        """Handle high CPU usage"""
        actions = []
        
        try:
            # Find processes with high CPU
            high_cpu_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'username']):
                try:
                    if proc.info['cpu_percent'] > 50:
                        high_cpu_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            high_cpu_processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            if high_cpu_processes:
                # Try to nice (lower priority) the top process
                top_process = high_cpu_processes[0]
                
                # Don't kill system processes
                if top_process['username'] not in ['root', 'system']:
                    action = self._nice_process(top_process['pid'])
                    if action:
                        actions.append(action)
                
                logger.info(f"High CPU detected: Top process is {top_process['name']} "
                          f"using {top_process['cpu_percent']:.1f}% CPU")
        
        except Exception as e:
            logger.error(f"Error handling high CPU: {e}")
            actions.append({
                'action': 'handle_high_cpu',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        return actions
    
    def _handle_high_memory(self, metrics: Dict) -> List[Dict]:
        """Handle high memory usage"""
        actions = []
        
        try:
            # Clear system caches (Linux only)
            if psutil.LINUX:
                action = self._clear_cache()
                if action:
                    actions.append(action)
            
            # Find memory-heavy processes
            high_mem_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'username']):
                try:
                    if proc.info['memory_percent'] > 10:
                        high_mem_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            high_mem_processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            if high_mem_processes:
                logger.info(f"High memory detected: Top process is {high_mem_processes[0]['name']} "
                          f"using {high_mem_processes[0]['memory_percent']:.1f}% memory")
        
        except Exception as e:
            logger.error(f"Error handling high memory: {e}")
            actions.append({
                'action': 'handle_high_memory',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        return actions
    
    def _handle_disk_full(self, metrics: Dict) -> List[Dict]:
        """Handle disk full condition"""
        actions = []
        
        try:
            # Clean temporary files
            action = self._clean_temp_files()
            if action:
                actions.append(action)
            
            # Suggest log rotation (don't do it automatically to avoid data loss)
            actions.append({
                'action': 'suggest_log_rotation',
                'status': 'suggestion',
                'message': 'Consider rotating or compressing log files',
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error handling disk full: {e}")
            actions.append({
                'action': 'handle_disk_full',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        return actions
    
    def _handle_high_load(self) -> List[Dict]:
        """Handle high system load"""
        actions = []
        
        try:
            actions.append({
                'action': 'monitor_load',
                'status': 'monitoring',
                'message': 'High system load detected, monitoring situation',
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error handling high load: {e}")
        
        return actions
    
    def _nice_process(self, pid: int) -> Optional[Dict]:
        """
        Lower the priority of a process
        
        Args:
            pid: Process ID
            
        Returns:
            Action dictionary or None
        """
        try:
            proc = psutil.Process(pid)
            original_nice = proc.nice()
            
            # Increase nice value (lower priority)
            new_nice = min(original_nice + 5, 19)
            proc.nice(new_nice)
            
            logger.info(f"Lowered priority of process {pid} from {original_nice} to {new_nice}")
            
            return {
                'action': 'nice_process',
                'status': 'success',
                'pid': pid,
                'process_name': proc.name(),
                'original_priority': original_nice,
                'new_priority': new_nice,
                'timestamp': datetime.now().isoformat()
            }
        
        except psutil.AccessDenied:
            logger.warning(f"Access denied when trying to nice process {pid}")
            return {
                'action': 'nice_process',
                'status': 'access_denied',
                'pid': pid,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error nicing process {pid}: {e}")
            return None
    
    def _clear_cache(self) -> Optional[Dict]:
        """
        Clear system caches (Linux only, requires root)
        
        Returns:
            Action dictionary or None
        """
        try:
            # Note: This requires root privileges
            # sync; echo 3 > /proc/sys/vm/drop_caches
            
            # We'll just log that we would do this
            logger.info("Would clear system caches (requires root privileges)")
            
            return {
                'action': 'clear_cache',
                'status': 'simulated',
                'message': 'Cache clearing requires root privileges',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return None
    
    def _clean_temp_files(self) -> Optional[Dict]:
        """
        Clean temporary files
        
        Returns:
            Action dictionary or None
        """
        try:
            temp_dirs = ['/tmp', '/var/tmp']
            cleaned_size = 0
            file_count = 0
            
            for temp_dir in temp_dirs:
                if not Path(temp_dir).exists():
                    continue
                
                # Only clean files older than 7 days for safety
                for item in Path(temp_dir).iterdir():
                    try:
                        # Check if older than 7 days
                        if item.is_file() and (datetime.now().timestamp() - item.stat().st_mtime) > 7 * 24 * 3600:
                            size = item.stat().st_size
                            item.unlink()
                            cleaned_size += size
                            file_count += 1
                    except (PermissionError, FileNotFoundError):
                        pass
            
            cleaned_mb = cleaned_size / (1024 ** 2)
            
            logger.info(f"Cleaned {file_count} temporary files, freed {cleaned_mb:.2f} MB")
            
            return {
                'action': 'clean_temp_files',
                'status': 'success',
                'files_cleaned': file_count,
                'space_freed_mb': cleaned_mb,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
            return {
                'action': 'clean_temp_files',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def restart_service(self, service_name: str) -> Dict:
        """
        Restart a system service
        
        Args:
            service_name: Name of the service
            
        Returns:
            Action dictionary
        """
        try:
            # Try systemctl first (systemd)
            result = subprocess.run(
                ['systemctl', 'restart', service_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully restarted service: {service_name}")
                return {
                    'action': 'restart_service',
                    'status': 'success',
                    'service': service_name,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to restart service {service_name}: {result.stderr}")
                return {
                    'action': 'restart_service',
                    'status': 'failed',
                    'service': service_name,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout restarting service: {service_name}")
            return {
                'action': 'restart_service',
                'status': 'timeout',
                'service': service_name,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
            return {
                'action': 'restart_service',
                'status': 'error',
                'service': service_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _log_action(self, action: Dict):
        """Log healing action to history"""
        self.action_history.append(action)
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    def get_action_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent healing actions
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of recent actions
        """
        return self.action_history[-limit:]
