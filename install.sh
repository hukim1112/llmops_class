#!/bin/bash

# 1. System Package Installation
# PDF processing and multimodal features require system-level dependencies.
echo "============================================"
echo "📦 Checking system packages..."
echo "============================================"

if command -v pdftoppm >/dev/null 2>&1; then
    echo "✅ 'poppler-utils' is already installed."
else
    echo "poppler-utils not found. Attempting install..."
    if [ -f /etc/debian_version ]; then
        # apt-get update가 부차적인 경고 등으로 실패하더라도 install을 진행하기 위해 분리 및 || true 적용
        sudo apt-get update || true
        sudo apt-get install -y poppler-utils
    else
        echo "⚠️ Non-Debian based system detected. Please install 'poppler-utils' manually."
    fi
fi

# 2. Python Dependencies
echo ""
echo "============================================"
echo "🐍 Installing Python dependencies..."
echo "============================================"

# 적절한 pip 실행 파일 결정 (활성화된 venv -> 기본 venv -> 시스템 pip 순서)
if [ -n "$VIRTUAL_ENV" ]; then
    echo "💡 Active virtual environment detected: $VIRTUAL_ENV"
    PIP_CMD="pip"
elif [ -d "$HOME/env_langchain_123" ]; then
    echo "💡 Found default virtual environment at ~/env_langchain_123"
    PIP_CMD="$HOME/env_langchain_123/bin/pip"
else
    if command -v pip >/dev/null 2>&1; then
        PIP_CMD="pip"
    elif command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    else
        PIP_CMD="python3 -m pip"
    fi
fi

echo "Using pip command: $PIP_CMD"

# 기본 pip install 시도 후, 실패(externally-managed-environment 등) 시 --break-system-packages 적용
if $PIP_CMD install -r requirements.txt; then
    echo "✅ Dependencies installed successfully."
else
    echo "⚠️ Standard installation failed. Attempting with --break-system-packages..."
    $PIP_CMD install -r requirements.txt --break-system-packages
fi

# 3. Environment File (.env) Setup
echo ""
echo "============================================"
echo "⚙️ Configuring environment variables..."
echo "============================================"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_TEMPLATE="$SCRIPT_DIR/.env_template"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_TEMPLATE" ]; then
        echo "📄 Creating .env from template..."
        cp "$ENV_TEMPLATE" "$ENV_FILE"
    else
        echo "⚠️ .env_template not found. Creating a blank .env..."
        touch "$ENV_FILE"
    fi
fi

# Update PROJECT_ROOT in .env dynamically to the current project folder path
if grep -q "PROJECT_ROOT=" "$ENV_FILE"; then
    # Replace existing PROJECT_ROOT line
    sed -i "s|PROJECT_ROOT=.*|PROJECT_ROOT=$SCRIPT_DIR|g" "$ENV_FILE"
else
    # Append PROJECT_ROOT if it doesn't exist
    echo "PROJECT_ROOT=$SCRIPT_DIR" >> "$ENV_FILE"
fi

echo "✅ PROJECT_ROOT set to: $SCRIPT_DIR"

echo ""
echo "============================================"
echo "🎉 Setup Completed Successfully!"
echo "============================================"
echo "Please remember to fill in your API Keys inside the '.env' file:"
echo "- OPENAI_API_KEY"
echo "- TAVILY_API_KEY"
echo "- GOOGLE_API_KEY"
echo "- LANGSMITH_API_KEY"
