# Rusty Bridger

Cross-platform bridge between face tracking sources and [VTube Studio](https://github.com/DenchiSoft/VTubeStudio).  
Supports **iPhone** (via VTube Studio iOS app) and **webcam** (neural ONNX-based tracking).

> Alternative to [VBridger](https://store.steampowered.com/app/1898830/VBridger/) — free and open source.

---

## Download

Go to [**Releases**](https://github.com/LakoMoor/rusty-bridger/releases/latest) and grab the installer for your platform:

| Platform | File |
|----------|------|
| macOS    | `RustyBridger-x.x.x-macos.dmg` |
| Linux    | `RustyBridger-x.x.x-linux-amd64.deb` |
| Windows  | `RustyBridger-x.x.x-windows-setup.exe` |

---

## Quick Start

1. Open **VTube Studio** on PC and make sure the API is enabled (port 8001).
2. Launch **Rusty Bridger**.
3. Go to the **Config** tab → load or create a transform config (`.json`).
4. Go to **Bridge** tab → choose source (iPhone or Webcam) → press **Connect**.
5. VTube Studio will show an authentication popup — accept it.

---

## Sources

### iPhone
Uses VTube Studio's iOS face tracking over local Wi-Fi.  
- Open VTube Studio on your iPhone.
- Enable *"Send data to PC"* in the app settings.
- Enter the iPhone's local IP in Rusty Bridger.

### Webcam (neural)
Uses two ONNX models downloaded automatically on first run (~3 MB total):
- **UltraFace RFB-320** — face detection
- **106-point MobileNetV1** — facial landmark detection

Models are saved to `~/.rusty-bridge/` and reused on subsequent launches.

---

## Transform Config

A `.json` file that defines how raw tracking values are mapped to VTube Studio parameters.  
You can edit it directly in the **Config** tab inside the app, or in any text editor.

### Format

```json
[
  {
    "name": "FaceAngleY",
    "func": "-HeadRotY * 1",
    "min": -40.0,
    "max": 40.0,
    "defaultValue": 0
  }
]
```

| Field          | Description |
|----------------|-------------|
| `name`         | VTube Studio parameter name. Built-in params are reused; unknown names create custom params. |
| `func`         | Math expression. Evaluated with [evalexpr](https://docs.rs/evalexpr/latest/evalexpr/). |
| `min` / `max`  | Parameter range sent to VTube Studio. |
| `defaultValue` | Value when face is not detected. |

### Available variables

#### Head

```
HeadRotX  HeadRotY  HeadRotZ
HeadPosX  HeadPosY  HeadPosZ
```

#### Eyes

```
EyeBlinkLeft   EyeBlinkRight
EyeLookDownLeft  EyeLookDownRight  EyeLookInLeft   EyeLookInRight
EyeLookOutLeft   EyeLookOutRight   EyeLookUpLeft   EyeLookUpRight
EyeSquintLeft    EyeSquintRight    EyeWideLeft     EyeWideRight
```

#### Mouth & Jaw

```
JawForward  JawLeft  JawOpen  JawRight
MouthClose  MouthDimpleLeft  MouthDimpleRight
MouthFrownLeft  MouthFrownRight  MouthFunnel
MouthLeft  MouthLowerDownLeft  MouthLowerDownRight
MouthPressLeft  MouthPressRight  MouthPucker  MouthRight
MouthRollLower  MouthRollUpper  MouthShrugLower  MouthShrugUpper
MouthSmileLeft  MouthSmileRight  MouthStretchLeft  MouthStretchRight
MouthUpperUpLeft  MouthUpperUpRight
```

#### Brows, Cheeks, Nose, Tongue

```
BrowDownLeft  BrowDownRight  BrowInnerUp  BrowOuterUpLeft  BrowOuterUpRight
CheekPuff  CheekSquintLeft  CheekSquintRight
NoseSneerLeft  NoseSneerRight
TongueOut
```

> **Webcam note:** webcam tracking provides a subset of the above (head rotation/position, eye blink, jaw open, mouth smile/frown, brow up). Full ARKit blendshapes require iPhone.

### Example config

See [`configs/`](configs/) for a ready-to-use example covering face angle, eye open/close, mouth, brows, and body movement.

---

## Building from source

```bash
git clone https://github.com/LakoMoor/rusty-bridger.git
cd rusty-bridger
cargo build --release -p rusty-bridge-ui
# Binary: target/release/rusty-bridge-ui
```

### Requirements

- Rust 1.78+
- **macOS**: Xcode Command Line Tools
- **Linux**: `libgtk-3-dev libv4l-dev libudev-dev`
- **Windows**: MSVC toolchain

### Creating installers

```bash
# macOS — run on macOS:
bash dist/macos/build_dmg.sh

# Linux — run on Linux:
bash dist/linux/build_deb.sh

# Windows — run on Windows:
dist\windows\build_exe.bat
```

Or push a `v*` tag to trigger the [GitHub Actions release workflow](.github/workflows/release.yml) which builds all three automatically.

---

## License

[MIT](LICENSE)
