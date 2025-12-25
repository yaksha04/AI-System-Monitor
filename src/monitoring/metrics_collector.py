"""
Metrics Collector Module
Collects system metrics using psutil
"""

import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects various system metrics"""
    
    def __init__(self):
        """Initialize the metrics collector"""
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_check_time = time.time()
        
    def collect_all_metrics(self) -> Dict:
        """
        Collect all system metrics
        
        Returns:
            Dictionary containing all metrics
        """
        try:
            current_time = time.time()
            time_delta = current_time - self.last_check_time
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'epoch_time': current_time,
            }
            
            # CPU Metrics
            metrics.update(self._collect_cpu_metrics())
            
            # Memory Metrics
            metrics.update(self._collect_memory_metrics())
            
            # Disk Metrics
            metrics.update(self._collect_disk_metrics(time_delta))
            
            # Network Metrics
            metrics.update(self._collect_network_metrics(time_delta))
            
            # Process Metrics
            metrics.update(self._collect_process_metrics())
            
            # Load Average (Unix only)
            metrics.update(self._collect_load_average())
            
            self.last_check_time = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}
    
    def _collect_cpu_metrics(self) -> Dict:
        """Collect CPU related metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'cpu_freq_min': cpu_freq.min if cpu_freq else 0,
                'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            return {
                'cpu_percent': 0,
                'cpu_count': 0,
                'cpu_freq_current': 0,
                'cpu_freq_min': 0,
                'cpu_freq_max': 0,
            }
    
    def _collect_memory_metrics(self) -> Dict:
        """Collect memory related metrics"""
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()
            
            return {
                'memory_total_mb': virtual_mem.total / (1024 ** 2),
                'memory_available_mb': virtual_mem.available / (1024 ** 2),
                'memory_used_mb': virtual_mem.used / (1024 ** 2),
                'memory_percent': virtual_mem.percent,
                'swap_total_mb': swap_mem.total / (1024 ** 2),
                'swap_used_mb': swap_mem.used / (1024 ** 2),
                'swap_percent': swap_mem.percent,
            }
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            return {
                'memory_total_mb': 0,
                'memory_available_mb': 0,
                'memory_used_mb': 0,
                'memory_percent': 0,
                'swap_total_mb': 0,
                'swap_used_mb': 0,
                'swap_percent': 0,
            }
    
    def _collect_disk_metrics(self, time_delta: float) -> Dict:
        """Collect disk related metrics"""
        try:
            disk_usage = psutil.disk_usage('/')
            current_disk_io = psutil.disk_io_counters()
            
            # Calculate rates (bytes per second)
            read_bytes = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta if time_delta > 0 else 0
            write_bytes = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta if time_delta > 0 else 0
            
            self.last_disk_io = current_disk_io
            
            return {
                'disk_total_gb': disk_usage.total / (1024 ** 3),
                'disk_used_gb': disk_usage.used / (1024 ** 3),
                'disk_free_gb': disk_usage.free / (1024 ** 3),
                'disk_percent': disk_usage.percent,
                'disk_io_read_mb': read_bytes / (1024 ** 2),
                'disk_io_write_mb': write_bytes / (1024 ** 2),
                'disk_io_read_count': current_disk_io.read_count,
                'disk_io_write_count': current_disk_io.write_count,
            }
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            return {
                'disk_total_gb': 0,
                'disk_used_gb': 0,
                'disk_free_gb': 0,
                'disk_percent': 0,
                'disk_io_read_mb': 0,
                'disk_io_write_mb': 0,
                'disk_io_read_count': 0,
                'disk_io_write_count': 0,
            }
    
    def _collect_network_metrics(self, time_delta: float) -> Dict:
        """Collect network related metrics"""
        try:
            current_net_io = psutil.net_io_counters()
            
            # Calculate rates (bytes per second)
            sent_bytes = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / time_delta if time_delta > 0 else 0
            recv_bytes = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / time_delta if time_delta > 0 else 0
            
            self.last_net_io = current_net_io
            
            return {
                'network_sent_mb': sent_bytes / (1024 ** 2),
                'network_recv_mb': recv_bytes / (1024 ** 2),
                'network_packets_sent': current_net_io.packets_sent,
                'network_packets_recv': current_net_io.packets_recv,
                'network_err_in': current_net_io.errin,
                'network_err_out': current_net_io.errout,
                'network_drop_in': current_net_io.dropin,
                'network_drop_out': current_net_io.dropout,
            }
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {
                'network_sent_mb': 0,
                'network_recv_mb': 0,
                'network_packets_sent': 0,
                'network_packets_recv': 0,
                'network_err_in': 0,
                'network_err_out': 0,
                'network_drop_in': 0,
                'network_drop_out': 0,
            }
    
    def _collect_process_metrics(self) -> Dict:
        """Collect process related metrics"""
        try:
            process_count = len(psutil.pids())
            
            # Get top processes by CPU and memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU
            top_cpu_processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5]
            
            # Sort by memory
            top_mem_processes = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]
            
            return {
                'process_count': process_count,
                'top_cpu_process': top_cpu_processes[0]['name'] if top_cpu_processes else 'N/A',
                'top_cpu_percent': top_cpu_processes[0]['cpu_percent'] if top_cpu_processes else 0,
                'top_mem_process': top_mem_processes[0]['name'] if top_mem_processes else 'N/A',
                'top_mem_percent': top_mem_processes[0]['memory_percent'] if top_mem_processes else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {
                'process_count': 0,
                'top_cpu_process': 'N/A',
                'top_cpu_percent': 0,
                'top_mem_process': 'N/A',
                'top_mem_percent': 0,
            }
    
    def _collect_load_average(self) -> Dict:
        """Collect system load average (Unix only)"""
        try:
            if hasattr(psutil, 'getloadavg'):
                load1, load5, load15 = psutil.getloadavg()
                return {
                    'load_average_1min': load1,
                    'load_average_5min': load5,
                    'load_average_15min': load15,
                }
            else:
                return {
                    'load_average_1min': 0,
                    'load_average_5min': 0,
                    'load_average_15min': 0,
                }
        except Exception as e:
            logger.error(f"Error collecting load average: {e}")
            return {
                'load_average_1min': 0,
                'load_average_5min': 0,
                'load_average_15min': 0,
            }
    
    def get_process_info(self, process_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific process
        
        Args:
            process_name: Name of the process
            
        Returns:
            Process information dictionary or None
        """
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                if proc.info['name'] == process_name:
                    return proc.info
            return None
        except Exception as e:
            logger.error(f"Error getting process info for {process_name}: {e}")
            return None
    
    def check_critical_processes(self, process_list: List[str]) -> Dict[str, bool]:
        """
        Check if critical processes are running
        
        Args:
            process_list: List of critical process names
            
        Returns:
            Dictionary mapping process name to running status
        """
        running_processes = {proc.name() for proc in psutil.process_iter(['name'])}
        return {proc: proc in running_processes for proc in process_list}