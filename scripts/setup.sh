#!/bin/bash

# AI System Monitor Setup Script
# This script sets up the entire project from scratch

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   AI-Powered Intelligent Linux System Monitor            ║"
echo "║              Setup Script v1.0                            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Check if Python is installed
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
print_info "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip is not installed. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi
print_success "pip is installed"

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv

if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment created and activated"
fi

# Install required packages
print_info "Installing Python dependencies..."
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p data/logs data/models data/metrics config tests scripts
print_success "Directory structure created"

# Create config file if it doesn't exist
if [ ! -f "config/config.yaml" ]; then
    print_info "Creating default configuration file..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
from main import SystemMonitorApp
app = SystemMonitorApp()
"
    print_success "Configuration file created"
else
    print_info "Configuration file already exists"
fi

# Create .env file for secrets
if [ ! -f ".env" ]; then
    print_info "Creating .env file for secrets..."
    cat > .env << EOF
# Email Configuration (Optional)
EMAIL_PASSWORD=your_email_password_here

# Slack Configuration (Optional)
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
EOF
    print_success ".env file created"
    print_info "Please edit .env file with your actual credentials"
else
    print_info ".env file already exists"
fi

# Set permissions
print_info "Setting permissions..."
chmod +x scripts/setup.sh
chmod +x src/main.py
print_success "Permissions set"

# Check for Docker
print_info "Checking Docker installation..."
if command -v docker &> /dev/null; then
    print_success "Docker is installed"
    
    read -p "Do you want to build Docker images? (y/n): " build_docker
    
    if [ "$build_docker" = "y" ] || [ "$build_docker" = "Y" ]; then
        print_info "Building Docker images..."
        docker-compose build
        print_success "Docker images built successfully"
    fi
else
    print_info "Docker is not installed (optional for containerized deployment)"
fi

# Run a test collection
print_info "Running test metric collection..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from monitoring.metrics_collector import MetricsCollector
collector = MetricsCollector()
metrics = collector.collect_all_metrics()
if metrics:
    print('Test collection successful!')
    print(f'CPU: {metrics.get(\"cpu_percent\", 0):.1f}%')
    print(f'Memory: {metrics.get(\"memory_percent\", 0):.1f}%')
    print(f'Disk: {metrics.get(\"disk_percent\", 0):.1f}%')
else:
    print('Test collection failed!')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Test collection passed"
else
    print_error "Test collection failed"
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              Setup Complete! 🎉                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Edit config/config.yaml to customize settings"
echo "2. Edit .env to add email/Slack credentials (optional)"
echo ""
echo "To run the monitor:"
echo "  python3 src/main.py"
echo ""
echo "To run the dashboard:"
echo "  streamlit run dashboard.py"
echo ""
echo "To run with Docker:"
echo "  docker-compose up -d"
echo ""
print_success "Happy monitoring! 🖥️"
