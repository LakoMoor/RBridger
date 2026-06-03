use std::{
    net::UdpSocket,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    time::Duration,
};

use log::{info, warn};
use serde::Deserialize;

use crate::vtsphone::{Cords, Shape, TrackingResponce};

/// ARKit/Perfect-Sync blendshape names as output by ExpressionApp.exe.
/// Index matches the `exp` array in each UDP packet.
const EXPR_NAMES: [&str; 53] = [
    "browDown_L",      "browDown_R",
    "browInnerUp_L",   "browInnerUp_R",
    "browOuterUp_L",   "browOuterUp_R",
    "cheekPuff_L",     "cheekPuff_R",
    "cheekSquint_L",   "cheekSquint_R",
    "eyeBlink_L",      "eyeBlink_R",
    "eyeLookDown_L",   "eyeLookDown_R",
    "eyeLookIn_L",     "eyeLookIn_R",
    "eyeLookOut_L",    "eyeLookOut_R",
    "eyeLookUp_L",     "eyeLookUp_R",
    "eyeSquint_L",     "eyeSquint_R",
    "eyeWide_L",       "eyeWide_R",
    "jawForward",      "jawLeft",
    "jawOpen",         "jawRight",
    "mouthClose",      "mouthDimple_L",
    "mouthDimple_R",   "mouthFrown_L",
    "mouthFrown_R",    "mouthFunnel",
    "mouthLeft",       "mouthLowerDown_L",
    "mouthLowerDown_R","mouthPress_L",
    "mouthPress_R",    "mouthPucker",
    "mouthRight",      "mouthRollLower",
    "mouthRollUpper",  "mouthShrugLower",
    "mouthShrugUpper", "mouthSmile_L",
    "mouthSmile_R",    "mouthStretch_L",
    "mouthStretch_R",  "mouthUpperUp_L",
    "mouthUpperUp_R",  "noseSneer_L",
    "noseSneer_R",
];

#[derive(Deserialize)]
struct ExprPacket {
    exp: Option<Vec<f32>>,
    rot: Option<Vec<f32>>,
    cnf: Option<f32>,
}

pub struct ExprAppTracker;

impl ExprAppTracker {
    /// Listens on `0.0.0.0:{port}` for UDP packets from ExpressionApp.exe
    /// (VTube Studio NVIDIA or MediaPipe Webcam Tracker DLC).
    /// Default port broadcast by ExpressionApp is 9140.
    pub fn run(port: u16, sender: Sender<TrackingResponce>, active: Arc<AtomicBool>) {
        let addr = format!("0.0.0.0:{port}");
        let sock = match UdpSocket::bind(&addr) {
            Ok(s) => s,
            Err(e) => { warn!("ExprApp: cannot bind {addr}: {e}"); return; }
        };
        sock.set_read_timeout(Some(Duration::from_millis(500))).ok();
        info!("ExprApp listener ready on {addr}");

        let mut ts: u64 = 0;
        let mut buf = vec![0u8; 65_536];

        while active.load(Ordering::Relaxed) {
            let n = match sock.recv(&mut buf) {
                Ok(n) => n,
                Err(_) => continue,
            };

            // ExpressionApp appends a trailing NUL — strip it before parsing.
            let end = if buf[..n].last() == Some(&0) { n - 1 } else { n };
            let pkt: ExprPacket = match serde_json::from_slice(&buf[..end]) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let cnf       = pkt.cnf.unwrap_or(0.0);
            let exp        = pkt.exp.unwrap_or_default();
            let face_found = cnf > 5.0 && !exp.is_empty();

            let blend_shapes = EXPR_NAMES
                .iter()
                .enumerate()
                .filter_map(|(i, name)| {
                    exp.get(i).map(|&v| Shape { k: (*name).to_string(), v: v as f64 })
                })
                .collect();

            let (pitch, yaw, roll) = pkt
                .rot
                .as_deref()
                .filter(|r| r.len() >= 4)
                .map(|r| quat_to_euler(r[0], r[1], r[2], r[3]))
                .unwrap_or((0.0, 0.0, 0.0));

            let _ = sender.send(TrackingResponce {
                timestamp: ts,
                hotkey: 0,
                face_found,
                rotation: Cords { x: pitch, y: yaw, z: roll },
                position: Cords { x: 0.0, y: 0.0, z: 0.0 },
                eye_left: Cords { x: 0.0, y: 0.0, z: 0.0 },
                blend_shapes,
            });
            ts += 1;
        }
    }
}

/// Quaternion (x,y,z,w) → (pitch, yaw, roll) in degrees.
fn quat_to_euler(x: f32, y: f32, z: f32, w: f32) -> (f64, f64, f64) {
    let sinr = 2.0 * (w * x + y * z);
    let cosr = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr.atan2(cosr).to_degrees() as f64;

    let sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
    let pitch = sinp.asin().to_degrees() as f64;

    let siny = 2.0 * (w * z + x * y);
    let cosy = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny.atan2(cosy).to_degrees() as f64;

    (pitch, yaw, roll)
}
