#!/usr/bin/env bash
# deploy/setup_ec2.sh
#
# Provision an Amazon Linux 2 / Ubuntu 22.04 EC2 instance to run ai-quant-lab.
#
# Usage
# -----
#   1. SSH into the instance:
#        ssh -i your-key.pem ec2-user@<instance-ip>
#
#   2. Upload the project (from your local machine, before or after SSH):
#        scp -i your-key.pem -r /path/to/ai-quant-lab ec2-user@<instance-ip>:~/ai-quant-lab
#
#   3. Run this script on the instance:
#        chmod +x ~/ai-quant-lab/deploy/setup_ec2.sh
#        ~/ai-quant-lab/deploy/setup_ec2.sh
#
# What it does
# ------------
#   - Installs Python 3.11+, pip, nginx
#   - Creates a dedicated 'quant' system user
#   - Creates a virtual-env at /opt/ai-quant-lab/venv
#   - Copies the project to /opt/ai-quant-lab/app
#   - Installs Python dependencies
#   - Installs the systemd service (ai_quant_lab.service)
#   - Installs the nginx reverse-proxy config
#   - Enables and starts both services

set -euo pipefail

APP_DIR=/opt/ai-quant-lab/app
VENV_DIR=/opt/ai-quant-lab/venv
SERVICE_NAME=ai_quant_lab
APP_USER=quant
APP_PORT=8000

# ── 0. Detect OS and install system packages ────────────────────────────────
echo "[setup] Detecting OS..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID=$ID
else
    OS_ID=unknown
fi

case "$OS_ID" in
    amzn)
        echo "[setup] Amazon Linux detected."
        sudo yum update -y
        sudo yum install -y python3.11 python3.11-pip nginx git
        PYTHON=python3.11
        ;;
    ubuntu|debian)
        echo "[setup] Ubuntu/Debian detected."
        sudo apt-get update -y
        sudo apt-get install -y python3.11 python3.11-venv python3-pip nginx git
        PYTHON=python3.11
        ;;
    *)
        echo "[setup] Unknown OS '$OS_ID' — assuming Python 3 is already installed."
        PYTHON=python3
        ;;
esac

# ── 1. Create app user ───────────────────────────────────────────────────────
echo "[setup] Creating system user '$APP_USER'..."
if ! id -u "$APP_USER" &>/dev/null; then
    sudo useradd --system --no-create-home --shell /sbin/nologin "$APP_USER"
fi

# ── 2. Set up app directory ──────────────────────────────────────────────────
echo "[setup] Copying project to $APP_DIR ..."
sudo mkdir -p "$APP_DIR"
# Copy all project files from the directory containing this script's parent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
sudo cp -r "$PROJECT_DIR"/. "$APP_DIR/"
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

# ── 3. Virtual environment ───────────────────────────────────────────────────
echo "[setup] Creating virtual environment at $VENV_DIR ..."
sudo mkdir -p "$VENV_DIR"
sudo "$PYTHON" -m venv "$VENV_DIR"
sudo chown -R "$APP_USER":"$APP_USER" "$VENV_DIR"

echo "[setup] Installing Python dependencies..."
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

# ── 4. .env file ─────────────────────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/env.example" ]; then
        echo "[setup] Copying env.example → .env (fill in your credentials!)"
        sudo cp "$APP_DIR/env.example" "$APP_DIR/.env"
        sudo chown "$APP_USER":"$APP_USER" "$APP_DIR/.env"
        sudo chmod 600 "$APP_DIR/.env"
    else
        echo "[setup] WARNING: no .env file found — create $APP_DIR/.env before starting the service."
    fi
fi

# ── 5. Systemd service ────────────────────────────────────────────────────────
echo "[setup] Installing systemd service..."
sudo cp "$SCRIPT_DIR/ai_quant_lab.service" /etc/systemd/system/
# Patch paths into the service file
sudo sed -i \
    -e "s|/opt/ai-quant-lab/app|$APP_DIR|g" \
    -e "s|/opt/ai-quant-lab/venv|$VENV_DIR|g" \
    -e "s|User=quant|User=$APP_USER|g" \
    /etc/systemd/system/ai_quant_lab.service

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
echo "[setup] Service '$SERVICE_NAME' started."

# ── 6. Nginx reverse proxy ────────────────────────────────────────────────────
echo "[setup] Installing nginx config..."
sudo cp "$SCRIPT_DIR/nginx.conf" /etc/nginx/conf.d/ai_quant_lab.conf
# Patch upstream port
sudo sed -i "s|127.0.0.1:8000|127.0.0.1:$APP_PORT|g" /etc/nginx/conf.d/ai_quant_lab.conf

sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx
echo "[setup] Nginx started."

# ── 7. Summary ────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<instance-ip>")
echo ""
echo "========================================"
echo " ai-quant-lab is live!"
echo "  API docs : http://$PUBLIC_IP/docs"
echo "  Logs     : sudo journalctl -u $SERVICE_NAME -f"
echo "  .env     : $APP_DIR/.env"
echo "========================================"
