use std::{
    net::UdpSocket,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    time,
};

use log::warn;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Cords {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Shape {
    pub k: String,
    pub v: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(rename_all = "PascalCase")]
pub struct TrackingResponce {
    pub timestamp: u64,
    pub hotkey: i16,
    pub face_found: bool,
    pub rotation: Cords,
    pub position: Cords,
    pub eye_left: Cords,
    pub blend_shapes: Vec<Shape>,
}

pub struct VtsPhone;

impl VtsPhone {
    pub fn run(ip: String, sender: Sender<TrackingResponce>, active: Arc<AtomicBool>) {
        let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
        let _ = socket.set_read_timeout(Some(time::Duration::new(2, 0)));
        let port = socket.local_addr().unwrap().port();

        let mut buf = [0; 4096];

        let request_traking: String = serde_json::json!({
            "messageType":"iOSTrackingDataRequest",
            "sentBy": "RustyBridge",
            "sendForSeconds": 10,
            "ports": [port]
        })
        .to_string();

        let mut next_time = time::Instant::now();
        let mut last_received = time::Instant::now();

        while active.load(Ordering::Relaxed) {
            if next_time <= time::Instant::now() {
                next_time = time::Instant::now() + time::Duration::from_secs(1);
                if let Err(e) = socket.send_to(request_traking.as_bytes(), format!("{:}:21412", ip)) {
                    warn!("Unable to request tracking data: {}", e);
                }
            }

            match socket.recv_from(&mut buf) {
                Ok((amt, _src)) => {
                    last_received = time::Instant::now();
                    match serde_json::from_slice::<TrackingResponce>(&buf[..amt]) {
                        Ok(data) => { let _ = sender.send(data); }
                        Err(e)   => { warn!("Deserialize tracking: {}", e); }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock
                       || e.kind() == std::io::ErrorKind::TimedOut => {
                    // Normal read timeout — check if phone has gone away
                    if last_received.elapsed() > time::Duration::from_secs(5) {
                        warn!("Phone disconnected (no data for 5 s)");
                        active.store(false, Ordering::Relaxed);
                        break;
                    }
                }
                Err(e) => {
                    warn!("Receive error: {}", e);
                }
            }
        }
    }
}
