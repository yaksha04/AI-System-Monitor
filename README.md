# AI-System-Monitor

An **AI-powered intelligent system monitoring platform** that performs **real-time system health monitoring**, **machine learning–based anomaly detection**, and **automated self-healing actions**, all wrapped inside a **lightweight, high-performance dashboard**.

This project demonstrates how **machine learning, system engineering, and automation** can be combined to build a **proactive observability and recovery system**.



## 📌 Project Highlights

- 🔍 **Real-time Monitoring** (CPU, Memory, Disk, Network, Load)
- 🤖 **Unsupervised ML Anomaly Detection** (Isolation Forest)
- 🔧 **Automated Self-Healing Actions**
- 📊 **Live Dashboard (Streamlit + HTML/CSS)**
- ⚡ **Low Overhead (<2% CPU, <200MB RAM)**
- 🐳 **Dockerized for Easy Deployment**
- 🧪 **Fully Tested (Unit, Integration, Performance)**



## 🧠 Motivation

Traditional monitoring tools rely on **static thresholds**, which often:
- Generate false alerts
- Miss gradual anomalies (memory leaks, performance drifts)
- Require manual intervention

This project solves those problems by introducing:
- **ML-driven anomaly detection**
- **Autonomous corrective actions**
- **Human-friendly visual monitoring**



## 🏗️ System Architecture
Metrics Collection
↓
Data Preprocessing
↓
ML Anomaly Detection
↓
Auto-Healing Engine
↓
Dashboard & Logs


Each module is **loosely coupled**, **thread-safe**, and **independently testable**.

---

## 🛠️ Tech Stack

### Languages
- **Python 3.8+**

### Core Libraries
- `psutil` – system metrics
- `scikit-learn` – ML models
- `pandas`, `numpy` – data processing
- `streamlit` – dashboard
- `pyyaml` – configuration management

### Tools
- Git & GitHub
- Docker & Docker Compose
- PyTest (testing)



## ⚙️ Features in Detail

### 📊 System Monitoring
- CPU usage
- Memory utilization
- Disk I/O rates
- Network throughput
- Load average
- Process count

### 🤖 Machine Learning
- Unsupervised anomaly detection
- Isolation Forest model
- Automatic retraining
- Feature scaling with StandardScaler
- Low-latency inference (~3 ms)

### 🔧 Auto-Healing
- CPU hog process priority reduction
- Memory cache cleanup
- Disk cleanup & log rotation
- Safe execution with:
  - Whitelisting
  - Rate limiting
  - Rollback support
  - Permission checks

### 🖥️ Dashboard
- Real-time updates
- Animated anomaly alerts
- Healing logs
- Clean HTML/CSS UI
- No Plotly (avoids duplicate ID issues)

---

## 📁 Project Structure

src/
├── main.py
├── monitoring/
│ ├── metrics_collector.py
│ └── system_monitor.py
├── ml/
│ └── anomaly_detector.py
├── healing/
│ └── auto_healer.py
├── notifications/
│ └── notifier.py
dashboard/
│ └── dashboard.py
config/
│ └── config.yaml
tests/
│ └── test_monitor.py
scripts/
│ └── setup.sh

## 🚀 Getting Started

### 1️⃣ Clone Repository
git clone https://github.com/yourusername/ai-system-monitor.git
cd ai-system-monitor

2️⃣ Setup Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3️⃣ Run Setup Script
chmod +x scripts/setup.sh
./scripts/setup.sh

▶️ Running the Application
CLI Monitoring Mode
python3 src/main.py

Dashboard Mode
streamlit run dashboard/dashboard.py
docker-compose up -d

🧪 Testing
Run all tests:
python3 tests/test_monitor.py -v
✔ 19 test cases
✔ 88% code coverage
✔ Unit + Integration + System tests

📈 Performance Results
Metric	Average	Peak
CPU Usage	1.8%	4.2%
Memory	142 MB	187 MB
Detection Latency	~8.6 sec	—
ML Inference	~3 ms	—


🧩 Inspiration & Learning
This project was inspired by my internship at Intello Labs, Gurugram, where I gained hands-on exposure to machine learning models and real-world ML applications. The learning experience played a key role in shaping the idea and design of this system.

🚀 Future Enhancements
Predictive failure forecasting

Distributed multi-node monitoring

Kubernetes & cloud integration

Advanced deep learning models

Centralized observability dashboard

👨‍💻 Author
YAKSHA
DevOps & ML Enthusiast
📍 India

