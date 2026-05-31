#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RustyBridge"
VERSION="0.1.0"
BINARY="rusty-bridge-ui"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ICNS="$WORKSPACE_ROOT/ui/resources/rb.icns"
RELEASE_BIN="$WORKSPACE_ROOT/target/release/$BINARY"
OUT_DIR="$WORKSPACE_ROOT/dist/out"
APP_DIR="$OUT_DIR/$APP_NAME.app"
DMG_PATH="$OUT_DIR/${APP_NAME}-${VERSION}-macos.dmg"

# Build release binary
echo "Building release..."
cd "$WORKSPACE_ROOT"
cargo build --release -p rusty-bridge-ui

# Assemble .app bundle
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$RELEASE_BIN" "$APP_DIR/Contents/MacOS/$APP_NAME"
[ -f "$ICNS" ] && cp "$ICNS" "$APP_DIR/Contents/Resources/AppIcon.icns"

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>         <string>com.ovrog.rustybridge</string>
  <key>CFBundleName</key>               <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>        <string>Rusty Bridge</string>
  <key>CFBundleExecutable</key>         <string>$APP_NAME</string>
  <key>CFBundleIconFile</key>           <string>AppIcon</string>
  <key>CFBundleVersion</key>            <string>$VERSION</string>
  <key>CFBundleShortVersionString</key> <string>$VERSION</string>
  <key>CFBundlePackageType</key>        <string>APPL</string>
  <key>NSCameraUsageDescription</key>   <string>Face tracking via webcam</string>
  <key>NSHighResolutionCapable</key>    <true/>
  <key>LSMinimumSystemVersion</key>     <string>11.0</string>
</dict>
</plist>
PLIST

# Sign the bundle (ad-hoc) so macOS allows launching
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true
/usr/bin/xattr -cr "$APP_DIR" 2>/dev/null || true

# Create DMG
mkdir -p "$OUT_DIR"
STAGING="$OUT_DIR/.dmg_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "$APP_DIR" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create -volname "$APP_NAME" \
  -srcfolder "$STAGING" \
  -ov -format UDZO \
  "$DMG_PATH"

rm -rf "$STAGING"
echo "DMG ready: $DMG_PATH"
