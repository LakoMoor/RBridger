use std::{
    io::Read as _,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use image::imageops;
use log::{info, warn};
use ndarray::{Array3, Array4};
use ort::{session::Session, value::TensorRef};

use crate::vtsphone::{Cords, Shape, TrackingResponce};

// ── Model URLs & paths ────────────────────────────────────────────────────────

const DET_URL: &str = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx";
const DET_PATH: &str = "face_det.onnx";

const LMK_URL: &str = "https://github.com/LakoMoor/RBridger/raw/master/models/face_landmarks_mediapipe.onnx";
const LMK_PATH: &str = "face_lmk_mp.onnx";

const BS_URL: &str = "https://github.com/LakoMoor/RBridger/raw/master/models/face_blendshapes_mediapipe.onnx";
const BS_PATH: &str = "face_bs_mp.onnx";

// ── MediaPipe constants ───────────────────────────────────────────────────────

// 146 landmark indices selected from 478-point mesh as input for blendshape model.
// Source: mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc
const BS_IDXS: [usize; 146] = [
      0,  1,  4,  5,  6,  7,  8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52,
     53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91,
     93, 95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,153,
    154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,
    197,234,246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,
    296,297,300,308,310,311,312,314,317,318,321,323,324,332,334,336,338,356,
    361,362,365,373,374,375,377,378,379,380,381,382,384,385,386,387,388,389,
    390,397,398,400,402,405,409,415,454,466,468,469,470,471,472,473,474,475,
    476,477,
];

// 52 blend shape names matching MediaPipe output order (index 0 = _neutral, skipped).
// Names are PascalCase to match VTube Studio ARKit parameter names.
const BS_NAMES: [&str; 52] = [
    "_neutral",
    "BrowDownLeft",  "BrowDownRight",  "BrowInnerUp",
    "BrowOuterUpLeft",  "BrowOuterUpRight",
    "CheekPuff",     "CheekSquintLeft",  "CheekSquintRight",
    "EyeBlinkLeft",  "EyeBlinkRight",
    "EyeLookDownLeft",  "EyeLookDownRight",
    "EyeLookInLeft",    "EyeLookInRight",
    "EyeLookOutLeft",   "EyeLookOutRight",
    "EyeLookUpLeft",    "EyeLookUpRight",
    "EyeSquintLeft",    "EyeSquintRight",
    "EyeWideLeft",      "EyeWideRight",
    "JawForward", "JawLeft", "JawOpen", "JawRight",
    "MouthClose",
    "MouthDimpleLeft",  "MouthDimpleRight",
    "MouthFrownLeft",   "MouthFrownRight",
    "MouthFunnel",  "MouthLeft",  "MouthRight",
    "MouthLowerDownLeft",  "MouthLowerDownRight",
    "MouthPressLeft",   "MouthPressRight",
    "MouthPucker",
    "MouthRollLower",  "MouthRollUpper",
    "MouthShrugLower", "MouthShrugUpper",
    "MouthSmileLeft",  "MouthSmileRight",
    "MouthStretchLeft","MouthStretchRight",
    "MouthUpperUpLeft","MouthUpperUpRight",
    "NoseSneerLeft",   "NoseSneerRight",
];

// ── Preview frame ─────────────────────────────────────────────────────────────

pub struct PreviewFrame {
    pub width:     u32,
    pub height:    u32,
    pub rgb:       Vec<u8>,
    pub landmarks: Vec<[f32; 2]>, // pixel coords in preview image space
    pub bbox:      Option<[f32; 4]>,
}

pub fn init_camera_permissions() {
    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|_| {});
}

// ── WebcamTracker ─────────────────────────────────────────────────────────────

pub struct WebcamTracker;

impl WebcamTracker {
    pub fn list_cameras() -> Vec<(u32, String)> {
        use nokhwa::utils::ApiBackend;
        nokhwa::query(ApiBackend::Auto)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| {
                let idx = c.index().as_index().ok()?;
                Some((idx as u32, c.human_name().to_string()))
            })
            .collect()
    }

    fn download(url: &str, path: &str) -> Result<(), String> {
        let p = PathBuf::from(path);
        if p.exists() && p.metadata().map(|m| m.len() > 4096).unwrap_or(false) {
            return Ok(());
        }
        info!("Downloading {}…", path);
        let resp = ureq::get(url).call().map_err(|e| format!("HTTP: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).map_err(|e| format!("Read: {e}"))?;
        std::fs::write(path, &buf).map_err(|e| format!("Write: {e}"))?;
        info!("Saved {} ({} KB)", path, buf.len() / 1024);
        Ok(())
    }

    pub fn run(
        camera_index: u32,
        sender:  Sender<TrackingResponce>,
        active:  Arc<AtomicBool>,
        preview: Arc<Mutex<Option<PreviewFrame>>>,
    ) {
        for (url, path) in [(DET_URL, DET_PATH), (LMK_URL, LMK_PATH), (BS_URL, BS_PATH)] {
            if let Err(e) = Self::download(url, path) {
                warn!("Model download {}: {}", path, e);
                return;
            }
        }

        let mut det = match Session::builder().and_then(|mut b| b.commit_from_file(DET_PATH)) {
            Ok(s) => s,
            Err(e) => { warn!("Load det: {}", e); return; }
        };
        let mut lmk = match Session::builder().and_then(|mut b| b.commit_from_file(LMK_PATH)) {
            Ok(s) => s,
            Err(e) => { warn!("Load lmk: {}", e); return; }
        };
        let mut bs = match Session::builder().and_then(|mut b| b.commit_from_file(BS_PATH)) {
            Ok(s) => s,
            Err(e) => { warn!("Load blendshapes: {}", e); return; }
        };

        info!("MediaPipe models loaded: det + landmark(478pt) + blendshapes(52)");

        use nokhwa::{
            pixel_format::RgbFormat,
            utils::{CameraIndex, RequestedFormat, RequestedFormatType},
            Camera,
        };
        let idx = CameraIndex::Index(camera_index);
        let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut cam = match Camera::new(idx, fmt) {
            Ok(c) => c,
            Err(e) => { warn!("Camera {}: {}", camera_index, e); return; }
        };
        if let Err(e) = cam.open_stream() { warn!("Stream: {}", e); return; }

        let mut ts: u64 = 0;
        while active.load(Ordering::Relaxed) {
            let frame = match cam.frame() {
                Ok(f) => f,
                Err(_) => { thread::sleep(Duration::from_millis(16)); continue; }
            };
            let img = match frame.decode_image::<RgbFormat>() {
                Ok(i) => i,
                Err(_) => continue,
            };
            let (iw, ih) = img.dimensions();

            let mut preview_lmks: Vec<[f32; 2]> = vec![];
            let mut preview_bbox: Option<[f32; 4]> = None;

            let tracking = match detect_face(&mut det, &img) {
                Some((bbox, score)) if score > 0.65 => {
                    preview_bbox = Some(bbox);
                    match detect_landmarks_mp(&mut lmk, &img, bbox) {
                        Some((lmks_norm, cx, cy, cw, ch)) => {
                            // Map normalized (in-crop) landmarks to image pixel coords for preview
                            preview_lmks = lmks_norm[..468].iter().map(|p| {
                                [p[0] * cw as f32 + cx as f32,
                                 p[1] * ch as f32 + cy as f32]
                            }).collect();

                            let shapes = compute_blendshapes(&mut bs, &lmks_norm);
                            let (pitch, yaw, roll, px, py, pz) =
                                head_pose(&lmks_norm, cx, cy, cw, ch, iw, ih);

                            TrackingResponce {
                                timestamp: ts, hotkey: 0, face_found: true,
                                rotation: Cords { x: pitch as f64, y: yaw as f64, z: roll as f64 },
                                position: Cords { x: px as f64,    y: py as f64,  z: pz as f64 },
                                eye_left: Cords { x: 0.0, y: 0.0, z: 0.0 },
                                blend_shapes: shapes,
                            }
                        }
                        None => no_face(ts),
                    }
                }
                _ => no_face(ts),
            };
            ts += 1;
            let _ = sender.send(tracking);

            // Write downscaled preview (non-blocking)
            if let Ok(mut guard) = preview.try_lock() {
                const MAX_W: u32 = 320;
                let (pw, ph, prgb, plmks, pbbox) = if iw > MAX_W {
                    let scale = MAX_W as f32 / iw as f32;
                    let pw = MAX_W;
                    let ph = (ih as f32 * scale) as u32;
                    let small = imageops::resize(&img, pw, ph, imageops::FilterType::Nearest);
                    let plmks = preview_lmks.iter()
                        .map(|p| [p[0] * scale, p[1] * scale])
                        .collect();
                    (pw, ph, small.into_raw(), plmks, preview_bbox)
                } else {
                    (iw, ih, img.into_raw(), preview_lmks, preview_bbox)
                };
                *guard = Some(PreviewFrame {
                    width: pw, height: ph, rgb: prgb,
                    landmarks: plmks, bbox: pbbox,
                });
            }
        }
        cam.stop_stream().ok();
    }
}

// ── UltraFace RFB-320 ─────────────────────────────────────────────────────────
// In:  [1,3,240,320] f32 (p-127)/128   Out: scores[1,4420,2] + boxes[1,4420,4]
fn detect_face(sess: &mut Session, img: &image::RgbImage) -> Option<([f32; 4], f32)> {
    const W: u32 = 320;
    const H: u32 = 240;
    let r = imageops::resize(img, W, H, imageops::FilterType::Nearest);
    let raw = r.as_raw();
    let n = (W * H) as usize;
    let mut data = vec![0f32; 3 * n];
    for i in 0..n {
        data[i]         = (raw[i * 3]     as f32 - 127.0) / 128.0;
        data[n + i]     = (raw[i * 3 + 1] as f32 - 127.0) / 128.0;
        data[2 * n + i] = (raw[i * 3 + 2] as f32 - 127.0) / 128.0;
    }
    let arr = Array4::from_shape_vec([1, 3, H as usize, W as usize], data).ok()?;
    let t = TensorRef::<f32>::from_array_view(arr.view()).ok()?;
    let outs = sess.run(ort::inputs![t]).ok()?;

    let (_, sv) = outs[0].try_extract_tensor::<f32>().ok()?;
    let (_, bv) = outs[1].try_extract_tensor::<f32>().ok()?;

    let (mut best, mut best_box) = (0f32, [0f32; 4]);
    for i in 0..4420 {
        let s = sv[i * 2 + 1];
        if s > best {
            best = s;
            best_box = [bv[i*4], bv[i*4+1], bv[i*4+2], bv[i*4+3]];
        }
    }
    Some((best_box, best))
}

// ── MediaPipe face landmark model ─────────────────────────────────────────────
// In:  [1, 256, 256, 3] NHWC, float [0,1]
// Out: [0] Identity [1,1,1,1434] — 478 landmarks × (x,y,z) normalized in crop [0,1]
//      [1] Identity_1 [1,1,1,1]  — face presence score
//
// Returns: (landmarks[478][3], crop_x1, crop_y1, crop_w, crop_h) in image pixels.
// Landmark x/y are normalized [0,1] within the crop; z is relative depth.
fn detect_landmarks_mp(
    sess: &mut Session,
    img:  &image::RgbImage,
    bbox: [f32; 4],
) -> Option<(Vec<[f32; 3]>, u32, u32, u32, u32)> {
    let (iw, ih) = img.dimensions();
    let m = 0.20_f32;
    let dx = (bbox[2] - bbox[0]) * m;
    let dy = (bbox[3] - bbox[1]) * m;
    let x1 = ((bbox[0] - dx) * iw as f32).max(0.0) as u32;
    let y1 = ((bbox[1] - dy) * ih as f32).max(0.0) as u32;
    let x2 = ((bbox[2] + dx) * iw as f32).min(iw as f32 - 1.0) as u32;
    let y2 = ((bbox[3] + dy) * ih as f32).min(ih as f32 - 1.0) as u32;
    if x2 <= x1 || y2 <= y1 { return None; }
    let cw = x2 - x1;
    let ch = y2 - y1;

    let crop = imageops::crop_imm(img, x1, y1, cw, ch).to_image();
    let rs = imageops::resize(&crop, 256, 256, imageops::FilterType::Nearest);
    let raw = rs.as_raw();

    // NHWC [1, 256, 256, 3] normalized [0, 1]
    let n_px = 256 * 256_usize;
    let mut data = vec![0f32; n_px * 3];
    for i in 0..n_px {
        data[i * 3]     = raw[i * 3]     as f32 / 255.0;
        data[i * 3 + 1] = raw[i * 3 + 1] as f32 / 255.0;
        data[i * 3 + 2] = raw[i * 3 + 2] as f32 / 255.0;
    }
    let arr = Array4::from_shape_vec([1, 256, 256, 3], data).ok()?;
    let t = TensorRef::<f32>::from_array_view(arr.view()).ok()?;
    let outs = sess.run(ort::inputs![t]).ok()?;

    // Check face presence
    let (_, presence) = outs[1].try_extract_tensor::<f32>().ok()?;
    if presence[0] < 0.5 { return None; }

    let (_, flat) = outs[0].try_extract_tensor::<f32>().ok()?;
    if flat.len() < 1434 { return None; }

    // 478 landmarks, each (x, y, z) normalized to [0,1] within the 256×256 crop
    let lmks: Vec<[f32; 3]> = (0..478)
        .map(|i| [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]])
        .collect();

    Some((lmks, x1, y1, cw, ch))
}

// ── MediaPipe blendshapes model ───────────────────────────────────────────────
// In:  [1, 146, 2] — 146 selected landmark (x, y) in crop-normalized [0,1]
// Out: [52] — blend shape coefficients [0,1]; index 0 is _neutral (skipped)
fn compute_blendshapes(sess: &mut Session, lmks: &[[f32; 3]]) -> Vec<Shape> {
    let mut bs_data = vec![0f32; 146 * 2];
    for (i, &idx) in BS_IDXS.iter().enumerate() {
        if idx < lmks.len() {
            bs_data[i * 2]     = lmks[idx][0];
            bs_data[i * 2 + 1] = lmks[idx][1];
        }
    }
    let arr = match Array3::from_shape_vec([1, 146, 2], bs_data) {
        Ok(a) => a,
        Err(_) => return vec![],
    };
    let t = match TensorRef::<f32>::from_array_view(arr.view()) {
        Ok(t) => t,
        Err(_) => return vec![],
    };
    let outs = match sess.run(ort::inputs![t]) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let (_, vals) = match outs[0].try_extract_tensor::<f32>() {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    // Skip index 0 (_neutral); map remaining 51 to named shapes
    (1..vals.len().min(52))
        .map(|i| Shape {
            k: BS_NAMES[i].to_string(),
            v: (vals[i] as f64).clamp(0.0, 1.0),
        })
        .collect()
}

// ── Head pose from MediaPipe 478 landmarks ────────────────────────────────────
// Key indices (MediaPipe face mesh topology):
//   nose tip:         4     left cheek:  234   right cheek:  454
//   chin:           152     left eye outer:  33  right eye outer: 263
fn head_pose(
    lmks: &[[f32; 3]],
    cx: u32, cy: u32, cw: u32, ch: u32,
    iw: u32, ih: u32,
) -> (f32, f32, f32, f32, f32, f32) {
    let fw = iw as f32;
    let fh = ih as f32;

    // Map crop-normalized [0,1] landmark to image pixel coords
    let px = |i: usize| lmks[i][0] * cw as f32 + cx as f32;
    let py = |i: usize| lmks[i][1] * ch as f32 + cy as f32;

    let nose_x  = px(4);
    let nose_y  = py(4);
    let l_eye_x = px(33);  let l_eye_y = py(33);
    let r_eye_x = px(263); let r_eye_y = py(263);
    let l_chk_x = px(234); let l_chk_y = py(234);
    let r_chk_x = px(454); let r_chk_y = py(454);
    let chin_y  = py(152);

    let face_cx = (l_chk_x + r_chk_x) / 2.0;
    let face_cy = (l_chk_y + r_chk_y) / 2.0;
    let face_w  = ((r_chk_x - l_chk_x).powi(2) + (r_chk_y - l_chk_y).powi(2)).sqrt();
    let eye_cy  = (l_eye_y + r_eye_y) / 2.0;
    let face_h  = (chin_y - eye_cy).max(1.0);

    let roll  = -(l_eye_y - r_eye_y).atan2(r_eye_x - l_eye_x).to_degrees();
    let yaw   = -(nose_x - face_cx) / face_w.max(1.0) * 55.0;
    let pitch = -(nose_y - (eye_cy + face_h * 0.3)) / face_h * 55.0;

    let pos_x = (face_cx - fw / 2.0) / fw * 20.0;
    let pos_y = (fh / 2.0 - face_cy) / fh * 20.0;
    let pos_z = ((fh * 0.3 / face_h) - 1.0) * 8.0;

    (pitch, yaw, roll, pos_x, pos_y, pos_z)
}

fn no_face(ts: u64) -> TrackingResponce {
    TrackingResponce {
        timestamp: ts, hotkey: 0, face_found: false,
        rotation: Cords { x: 0.0, y: 0.0, z: 0.0 },
        position: Cords { x: 0.0, y: 0.0, z: 0.0 },
        eye_left: Cords { x: 0.0, y: 0.0, z: 0.0 },
        blend_shapes: vec![],
    }
}
