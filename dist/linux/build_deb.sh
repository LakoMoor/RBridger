#!/usr/bin/env bash
set -euo pipefail

APP_NAME="rusty-bridge"
VERSION="0.1.0"
ARCH="amd64"
BINARY="rusty-bridge-ui"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RELEASE_BIN="$WORKSPACE_ROOT/target/release/$BINARY"
OUT_DIR="$WORKSPACE_ROOT/dist/out"
PKG_DIR="$OUT_DIR/${APP_NAME}_${VERSION}_${ARCH}"
DEB_PATH="$OUT_DIR/${APP_NAME}_${VERSION}_${ARCH}.deb"
ICON_SRC="$WORKSPACE_ROOT/ui/resources/rb128.png"

echo "Building release..."
cd "$WORKSPACE_ROOT"
cargo build --release -p rusty-bridge-ui

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/applications"
mkdir -p "$PKG_DIR/usr/share/pixmaps"

cp "$RELEASE_BIN" "$PKG_DIR/usr/bin/$APP_NAME"
[ -f "$ICON_SRC" ] && cp "$ICON_SRC" "$PKG_DIR/usr/share/pixmaps/rusty-bridge.png"

cat > "$PKG_DIR/DEBIAN/control" <<CTRL
Package: $APP_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: libc6 (>= 2.31)
Maintainer: ovROG <maintainer@example.com>
Description: Rusty Bridge - VTube Studio motion bridge
 Cross-platform bridge for VTube Studio with iPhone and webcam
 face tracking support. Includes built-in config editor.
CTRL

cat > "$PKG_DIR/usr/share/applications/rusty-bridge.desktop" <<DESKTOP
[Desktop Entry]
Name=Rusty Bridge
Comment=VTube Studio motion bridge
Exec=/usr/bin/$APP_NAME
Icon=rusty-bridge
Terminal=false
Type=Application
Categories=Utility;
DESKTOP

chmod 755 "$PKG_DIR/usr/bin/$APP_NAME"
dpkg-deb --build --root-owner-group "$PKG_DIR" "$DEB_PATH"
echo "DEB ready: $DEB_PATH"
