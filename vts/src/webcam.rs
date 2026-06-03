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
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::vtsphone::{Cords, Shape, TrackingResponce};

const DET_URL: &str = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx";
const LMK_URL: &str = "https://huggingface.co/kunkunlin1221/face-landmarks-2d-106_mbv1/resolve/main/coordinate_reg_mbv1_fp32.onnx";
const DET_PATH: &str = "face_det.onnx";
const LMK_PATH: &str = "face_lmk.onnx";

/// A single camera frame with detected face data, shared with the UI for preview.
pub struct PreviewFrame {
    pub width:     u32,
    pub height:    u32,
    pub rgb:       Vec<u8>,          // raw RGB pixels (width * height * 3)
    pub landmarks: Vec<[f32; 2]>,    // pixel coords in preview image space
    pub bbox:      Option<[f32; 4]>, // normalized [x1,y1,x2,y2] 0..1
}

pub fn init_camera_permissions() {
    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|_| {});
}

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
        let resp = ureq::get(url)
            .call()
            .map_err(|e| format!("HTTP: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("Read: {e}"))?;
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
        if let Err(e) = Self::download(DET_URL, DET_PATH) {
            warn!("Det model: {}", e);
            return;
        }
        if let Err(e) = Self::download(LMK_URL, LMK_PATH) {
            warn!("Lmk model: {}", e);
            return;
        }

        let mut det = match Session::builder().and_then(|mut b| b.commit_from_file(DET_PATH)) {
            Ok(s) => s,
            Err(e) => { warn!("Load det: {}", e); return; }
        };
        let mut lmk = match Session::builder().and_then(|mut b| b.commit_from_file(LMK_PATH)) {
            Ok(s) => s,
            Err(e) => { warn!("Load lmk: {}", e); return; }
        };

        info!("Models loaded: det={} lmk={}", DET_PATH, LMK_PATH);

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
        if let Err(e) = cam.open_stream() {
            warn!("Stream: {}", e); return;
        }

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

            // Run detection and collect data for both tracking and preview.
            let mut preview_lmks: Vec<[f32; 2]> = vec![];
            let mut preview_bbox: Option<[f32; 4]> = None;

            let tracking = match detect_face(&mut det, &img) {
                Some((bbox, score)) if score > 0.65 => {
                    preview_bbox = Some(bbox);
                    match detect_landmarks(&mut lmk, &img, bbox) {
                        Some(lmks) => {
                            preview_lmks = lmks.clone();
                            blendshapes(lmks, bbox, iw, ih, ts)
                        }
                        None => no_face(ts),
                    }
                }
                _ => no_face(ts),
            };
            ts += 1;
            let _ = sender.send(tracking);

            // Write downscaled preview frame (non-blocking).
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

// ── UltraFace RFB-320 ────────────────────────────────────────────────────────
// In:  [1,3,240,320] f32 normalised (p-127)/128
// Out: [0] scores[1,4420,2]  [1] boxes[1,4420,4]
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

// ── 106-point MobileNetV1 landmark model ────────────────────────────────────
// In:  [1,3,192,192] f32 normalised (p-127.5)/127.5  → [-1, 1]
// Out: [1,212] f32 interleaved (x0,y0,x1,y1,...) in [-1,1] relative to crop.
//      Convert: coord = (raw + 1) / 2 * crop_dim + crop_offset
fn detect_landmarks(sess: &mut Session, img: &image::RgbImage, bbox: [f32; 4]) -> Option<Vec<[f32; 2]>> {
    let (iw, ih) = img.dimensions();
    let m = 0.15_f32;
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
    const LW: u32 = 192;
    const LH: u32 = 192;
    let rs = imageops::resize(&crop, LW, LH, imageops::FilterType::Nearest);
    let raw = rs.as_raw();
    let n = (LW * LH) as usize;
    let mut data = vec![0f32; 3 * n];
    for i in 0..n {
        data[i]         = (raw[i * 3]     as f32 - 127.5) / 127.5;
        data[n + i]     = (raw[i * 3 + 1] as f32 - 127.5) / 127.5;
        data[2 * n + i] = (raw[i * 3 + 2] as f32 - 127.5) / 127.5;
    }
    let arr = Array4::from_shape_vec([1, 3, LH as usize, LW as usize], data).ok()?;
    let t = TensorRef::<f32>::from_array_view(arr.view()).ok()?;
    let outs = sess.run(ort::inputs![t]).ok()?;

    let (_, flat) = outs[0].try_extract_tensor::<f32>().ok()?;
    if flat.len() < 212 { return None; }

    // Output is interleaved [x0,y0,...] in [-1,1]; map back to pixel space
    Some(
        (0..106)
            .map(|i| [
                (flat[i * 2]     + 1.0) / 2.0 * cw as f32 + x1 as f32,
                (flat[i * 2 + 1] + 1.0) / 2.0 * ch as f32 + y1 as f32,
            ])
            .collect(),
    )
}

// ── Geometry → blendshapes ──────────────────────────────────────────────────
// InsightFace 2D-106 confirmed layout (from official markup image):
//  0-32   face contour (non-sequential oval)
//  33-42  left eyebrow  (subject's right, left in image)
//  43-51  left eye      (subject's right)
//  52-71  mouth         (outer 52-63, inner 64-71)
//  72-86  nose          (bridge 72-73, tip ~78, nostrils 79-86)
//  87-96  right eyebrow (subject's left, right in image)
//  97-105 right eye     (subject's left)
fn blendshapes(pts: Vec<[f32; 2]>, bbox: [f32; 4], iw: u32, ih: u32, ts: u64) -> TrackingResponce {
    let fw = iw as f32;
    let fh = ih as f32;
    let fcx = (bbox[0] + bbox[2]) / 2.0 * fw;
    let fcy = (bbox[1] + bbox[3]) / 2.0 * fh;
    let face_w = (bbox[2] - bbox[0]) * fw;
    let face_h = (bbox[3] - bbox[1]) * fh;

    let pos_x = (fcx - fw / 2.0) / fw * 20.0;
    let pos_y = (fh / 2.0 - fcy) / fh * 20.0;
    let pos_z = ((fh * 0.3 / face_h.max(1.0)) - 1.0) * 8.0;

    // Roll: angle between left ear (pt 1) and right ear (pt 17)
    let roll = if pts.len() > 17 {
        (pts[1][1] - pts[17][1]).atan2(pts[17][0] - pts[1][0]).to_degrees()
    } else { 0.0 };

    // Nose tip ≈ pt 78 (middle of nose range 72-86)
    let nose = pts.get(78).copied().unwrap_or([fcx, fcy]);
    let yaw   = -(nose[0] - fcx) / face_w.max(1.0) * 55.0;

    // Eye centres: left eye in image = 43-51, right eye in image = 97-105
    let right_ec = if pts.len() > 51 { midpt(pts[44], pts[49]) } else { [fcx, fcy] };
    let left_ec  = if pts.len() > 105 { midpt(pts[98], pts[103]) } else { [fcx, fcy] };
    let eye_y  = (right_ec[1] + left_ec[1]) / 2.0;
    let pitch  = -(nose[1] - (eye_y + pts[16][1]) / 2.0) / face_h.max(1.0) * 55.0;

    // Eye Aspect Ratio → blink
    // Left eye in image (subject right): corners 43/51, top 46, bot 49
    let r_blink = if pts.len() > 51 {
        (1.0 - ear(pts[43], pts[51], pts[46], pts[49])).clamp(0.0, 1.0)
    } else { 0.0 };
    // Right eye in image (subject left): corners 97/105, top 100, bot 103
    let l_blink = if pts.len() > 105 {
        (1.0 - ear(pts[97], pts[105], pts[100], pts[103])).clamp(0.0, 1.0)
    } else { 0.0 };

    // Jaw open: inner mouth top (pt 66) vs inner bottom (pt 70)
    let jaw_open = if pts.len() > 71 {
        (dist(pts[66], pts[70]) / face_h.max(1.0) * 6.0).clamp(0.0, 1.0)
    } else { 0.0 };

    // Smile: outer mouth corners (52, 58) vs centre bottom (61)
    let smile = if pts.len() > 62 {
        let corner_y = (pts[52][1] + pts[58][1]) / 2.0;
        ((pts[61][1] - corner_y) / face_h.max(1.0) * 10.0).clamp(-1.0, 1.0)
    } else { 0.0 };

    // Brows: left brow mid ≈ pt 37, right brow mid ≈ pt 91
    let brow_ref = (left_ec[1] + right_ec[1]) / 2.0;
    let lb = if pts.len() > 42 {
        ((brow_ref - pts[37][1]) / face_h.max(1.0) * 8.0 + 0.5).clamp(0.0, 1.0)
    } else { 0.5 };
    let rb = if pts.len() > 96 {
        ((brow_ref - pts[91][1]) / face_h.max(1.0) * 8.0 + 0.5).clamp(0.0, 1.0)
    } else { 0.5 };

    TrackingResponce {
        timestamp: ts, hotkey: 0, face_found: true,
        rotation: Cords { x: pitch as f64, y: yaw as f64, z: roll as f64 },
        position: Cords { x: pos_x as f64, y: pos_y as f64, z: pos_z as f64 },
        eye_left: Cords { x: 0.0, y: 0.0, z: 0.0 },
        blend_shapes: vec![
            Shape { k: "EyeBlinkLeft".into(),      v: l_blink as f64 },
            Shape { k: "EyeBlinkRight".into(),     v: r_blink as f64 },
            Shape { k: "JawOpen".into(),            v: jaw_open as f64 },
            Shape { k: "MouthSmileLeft".into(),    v: smile.max(0.0) as f64 },
            Shape { k: "MouthSmileRight".into(),   v: smile.max(0.0) as f64 },
            Shape { k: "MouthFrownLeft".into(),    v: (-smile).max(0.0) as f64 },
            Shape { k: "MouthFrownRight".into(),   v: (-smile).max(0.0) as f64 },
            Shape { k: "BrowOuterUpLeft".into(),   v: lb as f64 },
            Shape { k: "BrowOuterUpRight".into(),  v: rb as f64 },
        ],
    }
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

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2)).sqrt()
}
fn midpt(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0]+b[0])/2.0, (a[1]+b[1])/2.0]
}
fn ear(lc: [f32;2], rc: [f32;2], top: [f32;2], bot: [f32;2]) -> f32 {
    (dist(top, bot) / dist(lc, rc).max(0.001) / 0.28).clamp(0.0, 1.0)
}
