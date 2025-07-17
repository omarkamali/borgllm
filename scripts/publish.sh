#!/bin/bash

set -e

echo "🚀 Starting BorgLLM publish process..."

# 1. Check if git working directory is clean
echo "🔍 Checking for uncommitted changes..."
if ! git diff --quiet --exit-code;
then
  echo "❌ Error: Git working directory is dirty. Please commit or stash your changes before publishing."
  git status
  exit 1
fi

if ! git diff --cached --quiet --exit-code;
then
  echo "❌ Error: Staged changes detected. Please commit or unstage your changes before publishing."
  git status
  exit 1
fi

echo "✅ Git working directory is clean."

# 2. Get version from pyproject.toml
VERSION=$(grep 'version = "' pyproject.toml | head -1 | sed -E 's/.*version = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')

if [ -z "$VERSION" ]; then
  echo "❌ Error: Could not extract version from pyproject.toml"
  exit 1
fi

echo "🔖 Detected version: v$VERSION"

# 3. Check if tag exists for the current version
echo "🏷️ Checking for git tag v$VERSION..."
if ! git rev-parse "v$VERSION" >/dev/null 2>&1;
then
  echo "❌ Error: Git tag v$VERSION does not exist. Please create it before publishing."
  echo "Example: git tag v$VERSION && git push origin v$VERSION"
  exit 1
fi

echo "✅ Git tag v$VERSION exists."

# Proceed with hatch publish
echo "📦 Running hatch publish..."
hatch clean
hatch build
hatch publish

# 4. Create GitHub Release
echo "🚀 Creating GitHub Release v$VERSION..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null
then
    echo "❌ Error: GitHub CLI (gh) is not installed. Please install it to create a GitHub release."
    echo "       See https://cli.github.com/ for installation instructions."
    exit 1
fi

# Create the GitHub release
gh release create "v$VERSION" --generate-notes --title "BorgLLM v$VERSION"

echo "🎉 BorgLLM publish process completed successfully!"
echo "✨ GitHub Release v$VERSION created successfully!" 