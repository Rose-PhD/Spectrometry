#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "📦 Starting Python Package Build & Publish Process"
echo "=================================================="

# 1. Request the API Token securely (input is hidden while typing)
echo -n "🔑 Enter your API Token (will be hidden): "
read -s API_TOKEN
echo "" # Move to a new line after hidden input

if [ -z "$API_TOKEN" ]; then
    echo "❌ Error: Token cannot be empty. Aborting process."
    exit 1
fi

# 2. Clean up previous build distributions safely
echo "🧹 Cleaning up old build artifacts..."
rm -rf build dist *.egg-info src/*.egg-info

# 3. Build the new package distributions
echo "🏗️  Building your wheel and source distributions..."
python -m build

# 4. Inspect the wheel contents to verify `.pkl` inclusion
echo "🔍 Checking included .pkl files in the distribution..."
# Use || true to prevent script failure if grep finds no matches
unzip -l dist/*.whl | grep "pkl" || echo "⚠️  No .pkl files found in the wheel."

# 5. Check the distributions for common rendering / format issues
echo "📋 Running twine checks..."
twine check dist/*

# 6. Upload distributions using the provided token
echo "🚀 Uploading package to repository..."
# Adjust the username below based on where you are uploading:
# - For PyPI: use "__token__" as the user and the API token as the password
# - For other custom repositories, adjust accordingly
TWINE_USERNAME="__token__" TWINE_PASSWORD="$API_TOKEN" twine upload dist/*

echo "=================================================="
echo "✅ Build and Publish Workflow Completed Successfully!"
echo "=================================================="
