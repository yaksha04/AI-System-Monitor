"""
Notification Module
Sends alerts via email and Slack
"""

import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class Notifier:
    """Handles notifications for system alerts"""
    
    def __init__(self, config: Dict):
        """
        Initialize the notifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.notif_config = config['notifications']
        self.enabled = self.notif_config['enabled']
        
        # Email configuration
        self.email_config = self.notif_config.get('email', {})
        self.email_enabled = self.email_config.get('enabled', False)
        
        # Slack configuration
        self.slack_config = self.notif_config.get('slack', {})
        self.slack_enabled = self.slack_config.get('enabled', False)
        
        # Notification history
        self.notification_history = []
    
    def send_anomaly_alert(self, anomaly_details: Dict, metrics: Dict, healing_result: Dict) -> Dict:
        """
        Send alert about detected anomaly
        
        Args:
            anomaly_details: Anomaly detection details
            metrics: Current system metrics
            healing_result: Auto-healing results
            
        Returns:
            Notification result dictionary
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        # Prepare message
        subject = "🚨 System Anomaly Detected"
        message = self._format_anomaly_message(anomaly_details, metrics, healing_result)
        
        results = []
        
        # Send email
        if self.email_enabled:
            email_result = self._send_email(subject, message)
            results.append(email_result)
        
        # Send Slack
        if self.slack_enabled:
            slack_result = self._send_slack(subject, message, 'warning')
            results.append(slack_result)
        
        # Log notification
        self._log_notification({
            'type': 'anomaly_alert',
            'subject': subject,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return {
            'status': 'sent',
            'channels': len(results),
            'results': results
        }
    
    def send_threshold_alert(self, violations: List[Dict]) -> Dict:
        """
        Send alert about threshold violations
        
        Args:
            violations: List of threshold violations
            
        Returns:
            Notification result dictionary
        """
        if not self.enabled or not violations:
            return {'status': 'disabled' if not self.enabled else 'no_violations'}
        
        subject = f"⚠️ Threshold Alert: {len(violations)} violation(s)"
        message = self._format_threshold_message(violations)
        
        results = []
        
        if self.email_enabled:
            email_result = self._send_email(subject, message)
            results.append(email_result)
        
        if self.slack_enabled:
            slack_result = self._send_slack(subject, message, 'danger')
            results.append(slack_result)
        
        self._log_notification({
            'type': 'threshold_alert',
            'subject': subject,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return {
            'status': 'sent',
            'channels': len(results),
            'results': results
        }
    
    def send_service_down_alert(self, service_issues: List[Dict]) -> Dict:
        """
        Send alert about services that are down
        
        Args:
            service_issues: List of service issues
            
        Returns:
            Notification result dictionary
        """
        if not self.enabled or not service_issues:
            return {'status': 'disabled' if not self.enabled else 'no_issues'}
        
        subject = f"🔴 Service Down: {len(service_issues)} service(s)"
        message = self._format_service_message(service_issues)
        
        results = []
        
        if self.email_enabled:
            email_result = self._send_email(subject, message)
            results.append(email_result)
        
        if self.slack_enabled:
            slack_result = self._send_slack(subject, message, 'danger')
            results.append(slack_result)
        
        self._log_notification({
            'type': 'service_down_alert',
            'subject': subject,
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return {
            'status': 'sent',
            'channels': len(results),
            'results': results
        }
    
    def _format_anomaly_message(self, anomaly_details: Dict, metrics: Dict, healing_result: Dict) -> str:
        """Format anomaly alert message"""
        msg = "System Anomaly Detected\n"
        msg += "=" * 50 + "\n\n"
        
        msg += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"Anomaly Score: {anomaly_details.get('anomaly_score', 0):.4f}\n\n"
        
        msg += "Current System Metrics:\n"
        msg += f"  - CPU Usage: {metrics.get('cpu_percent', 0):.1f}%\n"
        msg += f"  - Memory Usage: {metrics.get('memory_percent', 0):.1f}%\n"
        msg += f"  - Disk Usage: {metrics.get('disk_percent', 0):.1f}%\n"
        msg += f"  - Load Average: {metrics.get('load_average_1min', 0):.2f}\n\n"
        
        if 'anomalous_features' in anomaly_details and anomaly_details['anomalous_features']:
            msg += "Anomalous Features:\n"
            for feature in anomaly_details['anomalous_features']:
                msg += f"  - {feature['feature']}: {feature['value']:.2f} ({feature['reason']})\n"
            msg += "\n"
        
        if healing_result.get('actions'):
            msg += "Auto-Healing Actions Taken:\n"
            for action in healing_result['actions']:
                msg += f"  - {action['action']}: {action['status']}\n"
        else:
            msg += "No auto-healing actions were required.\n"
        
        return msg
    
    def _format_threshold_message(self, violations: List[Dict]) -> str:
        """Format threshold alert message"""
        msg = "Threshold Violations Detected\n"
        msg += "=" * 50 + "\n\n"
        
        msg += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for violation in violations:
            msg += f"⚠️ {violation['description']}:\n"
            msg += f"  Current: {violation['current_value']:.1f}%\n"
            msg += f"  Threshold: {violation['threshold']:.1f}%\n"
            msg += f"  Severity: {violation['severity'].upper()}\n\n"
        
        return msg
    
    def _format_service_message(self, service_issues: List[Dict]) -> str:
        """Format service down alert message"""
        msg = "Critical Services Down\n"
        msg += "=" * 50 + "\n\n"
        
        msg += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for issue in service_issues:
            msg += f"🔴 Service: {issue['process']}\n"
            msg += f"  Status: {issue['status']}\n"
            msg += f"  Severity: {issue['severity'].upper()}\n\n"
        
        return msg
    
    def _send_email(self, subject: str, message: str) -> Dict:
        """
        Send email notification
        
        Args:
            subject: Email subject
            message: Email message
            
        Returns:
            Result dictionary
        """
        try:
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port')
            from_email = self.email_config.get('from_email')
            to_emails = self.email_config.get('to_emails', [])
            password = os.getenv('EMAIL_PASSWORD', self.email_config.get('password', ''))
            
            if not all([smtp_server, smtp_port, from_email, to_emails, password]):
                logger.warning("Email configuration incomplete")
                return {
                    'channel': 'email',
                    'status': 'error',
                    'error': 'incomplete_configuration'
                }
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(from_email, password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {len(to_emails)} recipient(s)")
            
            return {
                'channel': 'email',
                'status': 'success',
                'recipients': len(to_emails)
            }
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                'channel': 'email',
                'status': 'error',
                'error': str(e)
            }
    
    def _send_slack(self, title: str, message: str, color: str = 'warning') -> Dict:
        """
        Send Slack notification
        
        Args:
            title: Message title
            message: Message content
            color: Message color (good, warning, danger)
            
        Returns:
            Result dictionary
        """
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL', self.slack_config.get('webhook_url', ''))
            
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return {
                    'channel': 'slack',
                    'status': 'error',
                    'error': 'webhook_not_configured'
                }
            
            # Prepare payload
            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': title,
                        'text': message,
                        'footer': 'AI System Monitor',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
                return {
                    'channel': 'slack',
                    'status': 'success'
                }
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
                return {
                    'channel': 'slack',
                    'status': 'error',
                    'error': f'HTTP {response.status_code}'
                }
        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return {
                'channel': 'slack',
                'status': 'error',
                'error': str(e)
            }
    
    def _log_notification(self, notification: Dict):
        """Log notification to history"""
        self.notification_history.append(notification)
        
        # Keep only last 50 notifications
        if len(self.notification_history) > 50:
            self.notification_history = self.notification_history[-50:]
    
    def get_notification_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent notifications
        
        Args:
            limit: Maximum number of notifications to return
            
        Returns:
            List of recent notifications
        """
        return self.notification_history[-limit:]
    
    def test_notifications(self) -> Dict:
        """
        Test notification channels
        
        Returns:
            Test results dictionary
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'channels': []
        }
        
        test_subject = "🧪 Test Notification - AI System Monitor"
        test_message = "This is a test notification from AI System Monitor.\n\n"
        test_message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        test_message += "If you received this, notifications are working correctly!"
        
        if self.email_enabled:
            email_result = self._send_email(test_subject, test_message)
            results['channels'].append(email_result)
        
        if self.slack_enabled:
            slack_result = self._send_slack(test_subject, test_message, 'good')
            results['channels'].append(slack_result)
        
        if not self.email_enabled and not self.slack_enabled:
            results['status'] = 'no_channels_enabled'
        else:
            successful = sum(1 for c in results['channels'] if c['status'] == 'success')
            results['status'] = 'success' if successful > 0 else 'failed'
            results['successful_channels'] = successful
        
        return results
