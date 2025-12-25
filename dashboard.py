"""
Streamlit Dashboard for AI System Monitor
Real-time visualization and monitoring interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from monitoring.system_monitor import SystemMonitor
from ml.anomaly_detector import AnomalyDetector
from healing.auto_healer import AutoHealer
from notifications.notifier import Notifier


# Page configuration
st.set_page_config(
    page_title="AI System Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .anomaly-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-alert {
        background-color: #00cc00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_components():
    """Initialize monitoring components"""
    config = load_config()
    monitor = SystemMonitor(config)
    detector = AnomalyDetector(config)
    healer = AutoHealer(config)
    notifier = Notifier(config)
    return config, monitor, detector, healer, notifier


def create_gauge_chart(value, title, max_value=100, threshold=80):
    """Create a gauge chart for metrics"""
    color = "green" if value < threshold else "orange" if value < threshold + 10 else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'lightgreen'},
                {'range': [threshold, threshold + 10], 'color': 'lightyellow'},
                {'range': [threshold + 10, max_value], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_time_series_chart(df, columns, title):
    """Create time series chart"""
    fig = go.Figure()
    
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[col],
                name=col.replace('_', ' ').title(),
                mode='lines'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig


def main():
    """Main dashboard function"""
    st.title("🖥️ AI-Powered System Monitor & Auto-Healer")
    st.markdown("Real-time system monitoring with AI-powered anomaly detection and automated healing")
    
    # Initialize components
    config, monitor, detector, healer, notifier = initialize_components()
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Refresh rate
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 30, 5)
    
    # Manual actions
    st.sidebar.markdown("### Manual Actions")
    if st.sidebar.button("🔄 Collect Metrics Now"):
        monitor.collect_and_store_metrics()
        st.sidebar.success("Metrics collected!")
    
    if st.sidebar.button("🧠 Train Model"):
        ml_data = monitor.get_metrics_for_ml(config['ml']['features'])
        if detector.train(ml_data):
            st.sidebar.success("Model trained successfully!")
        else:
            st.sidebar.error("Training failed!")
    
    if st.sidebar.button("📧 Test Notifications"):
        result = notifier.test_notifications()
        if result.get('status') == 'success':
            st.sidebar.success("Notifications sent!")
        else:
            st.sidebar.warning("No channels enabled or test failed")
    
    # Main content
    placeholder = st.empty()
    
    # Auto-refresh loop
    while True:
        with placeholder.container():
            # Collect current metrics
            current_metrics = monitor.collect_and_store_metrics()
            
            if not current_metrics:
                st.error("Failed to collect metrics")
                time.sleep(refresh_rate)
                continue
            
            # Detection
            is_anomaly = False
            anomaly_details = {}
            if detector.is_trained:
                is_anomaly, score, anomaly_details = detector.detect(current_metrics)
            
            # Top metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_val = current_metrics.get('cpu_percent', 0)
                st.metric(
                    label="🔵 CPU Usage",
                    value=f"{cpu_val:.1f}%",
                    delta=f"{cpu_val - 50:.1f}%" if cpu_val > 50 else f"{cpu_val:.1f}%"
                )
            
            with col2:
                mem_val = current_metrics.get('memory_percent', 0)
                st.metric(
                    label="🟢 Memory Usage",
                    value=f"{mem_val:.1f}%",
                    delta=f"{mem_val - 50:.1f}%" if mem_val > 50 else f"{mem_val:.1f}%"
                )
            
            with col3:
                disk_val = current_metrics.get('disk_percent', 0)
                st.metric(
                    label="🟡 Disk Usage",
                    value=f"{disk_val:.1f}%",
                    delta=f"{disk_val - 50:.1f}%" if disk_val > 50 else f"{disk_val:.1f}%"
                )
            
            with col4:
                load_val = current_metrics.get('load_average_1min', 0)
                st.metric(
                    label="⚡ Load Average (1m)",
                    value=f"{load_val:.2f}",
                    delta=f"{load_val - 1:.2f}"
                )
            
            # Anomaly alert
            if is_anomaly:
                st.markdown(f"""
                <div class="anomaly-alert">
                    <h3>🚨 ANOMALY DETECTED!</h3>
                    <p>Anomaly Score: {anomaly_details.get('anomaly_score', 0):.4f}</p>
                    <p>Timestamp: {anomaly_details.get('timestamp', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-healing
                healing_result = healer.heal(anomaly_details, current_metrics)
                if healing_result['action_count'] > 0:
                    st.success(f"🔧 Auto-healing performed {healing_result['action_count']} action(s)")
                    
                    with st.expander("View Healing Actions"):
                        for action in healing_result['actions']:
                            st.write(f"- **{action['action']}**: {action['status']}")
            
            # Gauge charts
            st.markdown("### 📊 Real-time Metrics")
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            
            with gauge_col1:
                fig_cpu = create_gauge_chart(
                    current_metrics.get('cpu_percent', 0),
                    "CPU Usage (%)",
                    threshold=config['monitoring']['thresholds']['cpu_percent']
                )
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with gauge_col2:
                fig_mem = create_gauge_chart(
                    current_metrics.get('memory_percent', 0),
                    "Memory Usage (%)",
                    threshold=config['monitoring']['thresholds']['memory_percent']
                )
                st.plotly_chart(fig_mem, use_container_width=True)
            
            with gauge_col3:
                fig_disk = create_gauge_chart(
                    current_metrics.get('disk_percent', 0),
                    "Disk Usage (%)",
                    threshold=config['monitoring']['thresholds']['disk_percent']
                )
                st.plotly_chart(fig_disk, use_container_width=True)
            
            # Historical charts
            st.markdown("### 📈 Historical Trends")
            
            # Get recent data
            recent_df = monitor.get_recent_metrics(minutes=60)
            
            if len(recent_df) > 0:
                # Convert timestamp
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
                
                # CPU and Memory chart
                fig_cpu_mem = create_time_series_chart(
                    recent_df,
                    ['cpu_percent', 'memory_percent'],
                    "CPU & Memory Usage Over Time"
                )
                st.plotly_chart(fig_cpu_mem, use_container_width=True)
                
                # Network chart
                fig_network = create_time_series_chart(
                    recent_df,
                    ['network_sent_mb', 'network_recv_mb'],
                    "Network I/O Over Time (MB/s)"
                )
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Disk I/O chart
                fig_disk_io = create_time_series_chart(
                    recent_df,
                    ['disk_io_read_mb', 'disk_io_write_mb'],
                    "Disk I/O Over Time (MB/s)"
                )
                st.plotly_chart(fig_disk_io, use_container_width=True)
            else:
                st.info("Collecting historical data... Please wait.")
            
            # System Information
            st.markdown("### 💻 System Information")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown("**Current Status:**")
                st.write(f"- Processes: {current_metrics.get('process_count', 0)}")
                st.write(f"- CPU Cores: {current_metrics.get('cpu_count', 0)}")
                st.write(f"- Memory Available: {current_metrics.get('memory_available_mb', 0):.0f} MB")
                st.write(f"- Disk Free: {current_metrics.get('disk_free_gb', 0):.1f} GB")
            
            with info_col2:
                st.markdown("**Top Processes:**")
                st.write(f"- Top CPU: {current_metrics.get('top_cpu_process', 'N/A')} "
                        f"({current_metrics.get('top_cpu_percent', 0):.1f}%)")
                st.write(f"- Top Memory: {current_metrics.get('top_mem_process', 'N/A')} "
                        f"({current_metrics.get('top_mem_percent', 0):.1f}%)")
            
            # ML Model Status
            st.markdown("### 🧠 AI Model Status")
            
            ml_col1, ml_col2, ml_col3 = st.columns(3)
            
            with ml_col1:
                if detector.is_trained:
                    st.success("✅ Model Trained")
                    if detector.last_train_time:
                        st.write(f"Last trained: {detector.last_train_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.warning("⏳ Model Not Trained")
                    total_metrics = len(monitor.metrics_df)
                    training_window = config['ml']['training_window_minutes']
                    interval = config['monitoring']['interval_seconds']
                    needed = (training_window * 60) // interval
                    st.write(f"Collecting data: {total_metrics}/{needed}")
            
            with ml_col2:
                st.info(f"Model Type: {config['ml']['model_type'].replace('_', ' ').title()}")
                st.write(f"Features: {len(config['ml']['features'])}")
            
            with ml_col3:
                if detector.should_retrain():
                    st.warning("⚠️ Retraining Recommended")
                else:
                    st.success("✅ Model Up to Date")
            
            # Statistics
            st.markdown("### 📊 Statistics")
            stats = monitor.get_statistics()
            
            if stats:
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Total Metrics", stats.get('total_metrics', 0))
                    st.metric("Recent Metrics (1h)", stats.get('recent_metrics', 0))
                
                with stats_col2:
                    cpu_stats = stats.get('cpu', {})
                    st.write("**CPU Statistics:**")
                    st.write(f"- Average: {cpu_stats.get('average', 0):.1f}%")
                    st.write(f"- Max: {cpu_stats.get('max', 0):.1f}%")
                    st.write(f"- Min: {cpu_stats.get('min', 0):.1f}%")
                
                with stats_col3:
                    mem_stats = stats.get('memory', {})
                    st.write("**Memory Statistics:**")
                    st.write(f"- Average: {mem_stats.get('average', 0):.1f}%")
                    st.write(f"- Max: {mem_stats.get('max', 0):.1f}%")
                    st.write(f"- Min: {mem_stats.get('min', 0):.1f}%")
            
            # Recent Actions
            st.markdown("### 🔧 Recent Healing Actions")
            recent_actions = healer.get_action_history(limit=5)
            
            if recent_actions:
                for action in reversed(recent_actions):
                    status_color = "🟢" if action['status'] == 'success' else "🔴"
                    st.write(f"{status_color} **{action['action']}** - {action['status']} - "
                            f"{action.get('timestamp', 'N/A')}")
            else:
                st.info("No healing actions taken yet")
            
            # Footer
            st.markdown("---")
            st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"Auto-refresh every {refresh_rate} seconds*")
        
        # Wait before next refresh
        time.sleep(refresh_rate)


if __name__ == '__main__':
    main()