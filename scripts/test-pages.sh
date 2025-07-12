#!/bin/bash

# Script to test GitHub Pages structure locally
# This replicates what the GitHub Actions workflow does

set -e

echo "🚀 Testing GitHub Pages structure locally..."

# Clean up any existing test site
rm -rf _site

# Build documentation
echo "📚 Building documentation..."
hatch run docs

# Create the site structure
echo "📁 Creating site structure..."
mkdir -p _site
mkdir -p _site/docs

# Copy files
echo "📄 Copying files..."
cp index.html _site/
cp -r docs/build/* _site/docs/

# Create .nojekyll file
touch _site/.nojekyll

# Show the structure
echo "📋 Site structure:"
ls -la _site/
echo ""
echo "📋 Docs structure:"
ls -la _site/docs/

echo ""
echo "✅ Site structure created successfully!"
echo "🌐 You can now serve the site locally with:"
echo "   cd _site && python -m http.server 8000"
echo "   Then visit: http://localhost:8000" 