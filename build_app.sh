#!/bin/bash
set -e

# Define directories
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_NAME="VISTA AI.app"
APP_DIR="$DIR/$APP_NAME"
BUILD_DIR="$DIR/build"
RUNTIME_CACHE="$BUILD_DIR/runtime-cache/python-arm64"
DEP_CACHE="$BUILD_DIR/dependency-cache"
MODELS_CACHE="$BUILD_DIR/models-cache"

echo "=== VISTA AI Offline Bundle Builder ==="
echo "Target: macOS Apple Silicon (arm64)"

# 1. Setup cache directories
mkdir -p "$RUNTIME_CACHE"
mkdir -p "$DEP_CACHE"
mkdir -p "$MODELS_CACHE"

# 2. Download Standalone Python (if not cached)
# Using astral-sh (formerly indygreg) python-build-standalone for Apple Silicon
PYTHON_TAR="cpython-3.10.13+20240224-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20240224/$PYTHON_TAR"

if [ ! -d "$RUNTIME_CACHE/python" ]; then
    echo "Downloading standalone Python for arm64..."
    curl -L "$PYTHON_URL" -o "$RUNTIME_CACHE/$PYTHON_TAR"
    echo "Extracting Python runtime..."
    tar -xzf "$RUNTIME_CACHE/$PYTHON_TAR" -C "$RUNTIME_CACHE"
fi

# 3. Create .app structure
echo "Creating application bundle structure..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources/app"
mkdir -p "$APP_DIR/Contents/Resources/app/bootstrap/models"
mkdir -p "$APP_DIR/Contents/Frameworks"

# Copy python runtime into Frameworks
echo "Embedding Python runtime..."
cp -a "$RUNTIME_CACHE/python" "$APP_DIR/Contents/Frameworks/"

# 4. Install Dependencies into embedded Python (with caching)
echo "Pre-packaging Python dependencies (Thick Client)..."
EMBEDDED_PYTHON="$APP_DIR/Contents/Frameworks/python/bin/python3"
EMBEDDED_PIP="$APP_DIR/Contents/Frameworks/python/bin/pip3"

# Upgrade pip and install wheel
"$EMBEDDED_PYTHON" -m pip install --upgrade pip setuptools wheel --cache-dir "$DEP_CACHE"

# Install API and Engine dependencies
echo "Installing API dependencies..."
"$EMBEDDED_PIP" install -r "$DIR/face_api/requirements.txt" --cache-dir "$DEP_CACHE"

echo "Installing Engine dependencies..."
"$EMBEDDED_PIP" install -r "$DIR/face_engine/requirements.txt" --cache-dir "$DEP_CACHE"

echo "Installing Vehicle Intelligence dependencies..."
"$EMBEDDED_PIP" install fast-plate-ocr ultralytics --cache-dir "$DEP_CACHE"

# Install InsightFace explicitly if missing
"$EMBEDDED_PIP" install insightface onnxruntime --cache-dir "$DEP_CACHE"

# 5. Cache and copy models
echo "Pre-packaging AI models..."
YOLO_MODEL="yolo26n-face.pt"
if [ ! -f "$MODELS_CACHE/$YOLO_MODEL" ]; then
    if [ -f "$DIR/face_engine/models/$YOLO_MODEL" ]; then
        cp "$DIR/face_engine/models/$YOLO_MODEL" "$MODELS_CACHE/"
    else
        echo "Downloading YOLO face model..."
        curl -L "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt" -o "$MODELS_CACHE/$YOLO_MODEL"
    fi
fi
cp "$MODELS_CACHE/$YOLO_MODEL" "$APP_DIR/Contents/Resources/app/bootstrap/models/"

# 6. Copy source code
echo "Bundling source code..."
cp "$DIR/supervisor_ui.py" "$APP_DIR/Contents/Resources/app/"
rsync -a --exclude='.venv*' --exclude='__pycache__' "$DIR/face_api" "$APP_DIR/Contents/Resources/app/"
rsync -a --exclude='.venv*' --exclude='__pycache__' "$DIR/face_engine" "$APP_DIR/Contents/Resources/app/"
rsync -a --exclude='.venv*' --exclude='__pycache__' "$DIR/edge" "$APP_DIR/Contents/Resources/app/"
if [ -d "$DIR/Vehicle Intelligence/SIH26187" ]; then
    rsync -a --exclude='.venv*' --exclude='__pycache__' --exclude='runs' --exclude='data' "$DIR/Vehicle Intelligence/SIH26187/" "$APP_DIR/Contents/Resources/app/vehicle_intelligence/"
elif [ -d "$DIR/vehicle_intelligence" ]; then
    rsync -a --exclude='.venv*' --exclude='__pycache__' --exclude='runs' --exclude='data' "$DIR/vehicle_intelligence/" "$APP_DIR/Contents/Resources/app/vehicle_intelligence/"
fi

# Remove __pycache__ and venvs from bundled source just in case
find "$APP_DIR/Contents/Resources/app" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$APP_DIR/Contents/Resources/app" -name ".venv*" -type d -exec rm -rf {} + 2>/dev/null || true

# 7. Create Info.plist
cat <<EOF > "$APP_DIR/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>vista_launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.vista.ai</string>
    <key>CFBundleName</key>
    <string>VISTA AI</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>VISTA AI requires camera access to perform live facial recognition and border surveillance.</string>
</dict>
</plist>
EOF

# 8. Create Launcher
cat <<EOF > "$APP_DIR/Contents/MacOS/vista_launcher"
#!/bin/bash
DIR="\$( cd "\$( dirname "\$0" )/.." && pwd )"
PYTHON_BIN="\$DIR/Frameworks/python/bin/python3"
APP_SCRIPT="\$DIR/Resources/app/supervisor_ui.py"

# Execute natively using embedded python
exec arch -arm64 "\$PYTHON_BIN" "\$APP_SCRIPT"
EOF
chmod +x "$APP_DIR/Contents/MacOS/vista_launcher"

# Calculate final footprint
echo "Build complete!"
echo "Final footprint of $APP_NAME:"
du -sh "$APP_DIR"
echo "You can now ZIP this .app and distribute it."

# 9. Final Verification
echo "Verifying application bundle..."
test -x "$APP_DIR/Contents/MacOS/vista_launcher" || {
    echo "ERROR: vista_launcher was not created"
    exit 1
}

test -x "$APP_DIR/Contents/Frameworks/python/bin/python3" || {
    echo "ERROR: bundled Python runtime missing"
    exit 1
}

test -f "$APP_DIR/Contents/Resources/app/supervisor_ui.py" || {
    echo "ERROR: supervisor_ui.py missing"
    exit 1
}

echo "Verification complete: Bundle looks structurally intact."
