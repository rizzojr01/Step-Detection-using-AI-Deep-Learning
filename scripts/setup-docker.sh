#!/bin/bash

# ════════════════════════════════════════════════
# ║           DOCKER SETUP SCRIPT                 ║
# ║    (Run this once to set up Docker login)     ║
# ════════════════════════════════════════════════

echo "🐳 Docker Hub Setup"
echo "=================="

# Check if .env already exists
if [[ -f ".env" ]]; then
    echo "✅ .env file already exists"
    echo "   You can edit it or run: source .env"
else
    # Create .env from template
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        echo "📋 Created .env file from template"
    else
        echo "# Docker Hub Configuration" > .env
        echo "DOCKER_TOKEN=your_docker_personal_access_token_here" >> .env
        echo "DOCKER_USER=bibektimilsina000" >> .env
        echo "📋 Created .env file"
    fi
fi

echo ""
echo "📝 Next steps:"
echo "1. Edit .env file and add your Docker Hub token:"
echo "   nano .env"
echo ""
echo "2. Load environment variables:"
echo "   source .env"
echo ""
echo "3. Build and push:"
echo "   ./build.sh"
echo ""
echo "🔑 Get your Docker Hub token from:"
echo "   https://hub.docker.com/settings/security"
