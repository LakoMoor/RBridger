#!/usr/bin/env bash
# Build installers for the current platform.
# macOS → DMG   Linux → DEB   Windows → run build_exe.bat instead
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
  Darwin) bash "$SCRIPT_DIR/macos/build_dmg.sh" ;;
  Linux)  bash "$SCRIPT_DIR/linux/build_deb.sh" ;;
  *)      echo "On Windows run: dist\\windows\\build_exe.bat"; exit 1 ;;
esac
