#!/bin/bash
# Bump PASsistant version
# Usage: ./scripts/bump_version.sh [major|minor|patch]
#
# Examples:
#   ./scripts/bump_version.sh patch   # 0.1.0 -> 0.1.1
#   ./scripts/bump_version.sh minor   # 0.1.0 -> 0.2.0
#   ./scripts/bump_version.sh major   # 0.1.0 -> 1.0.0

set -e

VERSION_FILE="src/__version__.py"

if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: $VERSION_FILE not found. Run this from the project root."
    exit 1
fi

# Read current version
CURRENT=$(grep -oP '__version__ = "\K[^"]+' "$VERSION_FILE")
echo "Current version: $CURRENT"

# Split into components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "${1:-patch}" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch|*)
        PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "New version:     $NEW_VERSION"

# Update version file
sed -i "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"

echo ""
echo "✅ Version bumped to $NEW_VERSION"
echo ""
echo "Commit with:"
echo "  git add -A && git commit -m \"Bump version to $NEW_VERSION\""
echo "  git tag v$NEW_VERSION"
echo "  git push && git push --tags"
