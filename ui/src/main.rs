use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc,
    },
    thread,
};

use eframe::egui;
use rusty_bridge_lib::{
    vtspc::{CalcFn, VtsPc},
    vtsphone::{TrackingResponce, VtsPhone},
    webcam::WebcamTracker,
};

fn app_dir() -> PathBuf {
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join(".rusty-bridge");
    let _ = fs::create_dir_all(&dir);
    dir
}

// ── Persist config ────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
struct Config {
    transform_path: Option<String>,
    ip: Option<String>,
    source: Option<u8>,
    camera_index: Option<u32>,
}

impl Config {
    fn path() -> PathBuf { app_dir().join("ui-cfg.json") }
    fn load() -> Self {
        fs::read_to_string(Self::path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    fn save(&self) {
        if let Ok(s) = serde_json::to_string(self) {
            let _ = fs::write(Self::path(), s);
        }
    }
}

// ── Enums ─────────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Source { IPhone, Webcam }

#[derive(PartialEq, Clone, Copy)]
enum Tab { Bridge, Config }

// ── Config editor ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct Editor {
    params:      Vec<CalcFn>,
    selected:    Option<usize>,
    buf_name:    String,
    buf_func:    String,
    buf_min:     String,
    buf_max:     String,
    buf_default: String,
    formula_ok:  Option<bool>,  // None = empty, Some(true/false) = valid/invalid
    name_dup:    bool,
    dirty:       bool,
    status:      String,
}

impl Editor {
    fn select(&mut self, idx: usize) {
        if idx >= self.params.len() { return; }
        let p = &self.params[idx];
        self.buf_name    = p.name.clone();
        self.buf_func    = p.func.clone();
        self.buf_min     = p.min.to_string();
        self.buf_max     = p.max.to_string();
        self.buf_default = p.default_value.to_string();
        self.selected    = Some(idx);
        self.validate_buffers();
    }

    fn validate_buffers(&mut self) {
        let f = self.buf_func.trim();
        self.formula_ok = if f.is_empty() {
            None
        } else {
            Some(evalexpr::build_operator_tree(f).is_ok())
        };
        let name = self.buf_name.trim();
        self.name_dup = self.params.iter().enumerate()
            .any(|(i, p)| Some(i) != self.selected && p.name == name);
    }

    fn apply_edit(&mut self) {
        let Some(idx) = self.selected else { return };
        let Some(p) = self.params.get_mut(idx) else { return };
        let name = self.buf_name.trim().to_string();
        if !name.is_empty() { p.name = name; }
        p.func          = self.buf_func.trim().to_string();
        p.min           = self.buf_min.parse().unwrap_or(p.min);
        p.max           = self.buf_max.parse().unwrap_or(p.max);
        p.default_value = self.buf_default.parse().unwrap_or(p.default_value);
        self.dirty = true;
    }

    fn add_param(&mut self) {
        self.apply_edit();
        let idx = self.params.len();
        self.params.push(CalcFn {
            name: format!("Param{}", idx + 1),
            func: "0".into(),
            min: -1.0, max: 1.0, default_value: 0.0,
        });
        self.dirty = true;
        self.select(idx);
    }

    fn delete_selected(&mut self) {
        let Some(idx) = self.selected else { return };
        self.params.remove(idx);
        self.dirty = true;
        self.selected = if self.params.is_empty() {
            None
        } else {
            let new = idx.min(self.params.len() - 1);
            self.select(new);
            Some(new)
        };
    }

    fn move_selected(&mut self, up: bool) {
        let Some(idx) = self.selected else { return };
        let new_idx = if up {
            if idx == 0 { return; }
            idx - 1
        } else {
            if idx + 1 >= self.params.len() { return; }
            idx + 1
        };
        self.apply_edit();
        self.params.swap(idx, new_idx);
        self.selected = Some(new_idx);
        self.dirty = true;
    }

    fn load_file(&mut self, path: &str) {
        match fs::read_to_string(path).and_then(|s| {
            serde_json::from_str::<Vec<CalcFn>>(&s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }) {
            Ok(params) => {
                self.params  = params;
                self.selected = None;
                self.buf_name.clear(); self.buf_func.clear();
                self.dirty  = false;
                self.status = format!("Loaded {} params", self.params.len());
            }
            Err(e) => self.status = format!("Load error: {e}"),
        }
    }

    fn save_file(&mut self, path: &str) {
        self.apply_edit();
        match serde_json::to_string_pretty(&self.params) {
            Ok(s) => match fs::write(path, s) {
                Ok(_)  => { self.dirty = false; self.status = "Saved".into(); }
                Err(e) => self.status = format!("Save error: {e}"),
            },
            Err(e) => self.status = format!("Serialize error: {e}"),
        }
    }
}

// ── App ───────────────────────────────────────────────────────────────────────

struct App {
    cfg:           Config,
    tab:           Tab,
    transform_path: String,
    phone_ip:      String,
    source:        Source,
    cameras:       Vec<(u32, String)>,
    selected_cam:  u32,
    active:        Arc<AtomicBool>,
    pending_path:  Option<Receiver<Option<String>>>,
    editor:        Editor,
}

impl App {
    fn new(_cc: &eframe::CreationContext) -> Self {
        let cfg = Config::load();
        let source = match cfg.source.unwrap_or(0) {
            1 => Source::Webcam,
            _ => Source::IPhone,
        };
        Self {
            transform_path: cfg.transform_path.clone().unwrap_or_default(),
            phone_ip:       cfg.ip.clone().unwrap_or_default(),
            selected_cam:   cfg.camera_index.unwrap_or(0),
            source,
            tab:    Tab::Bridge,
            cameras: WebcamTracker::list_cameras(),
            active:  Arc::new(AtomicBool::new(false)),
            pending_path: None,
            editor:  Editor::default(),
            cfg,
        }
    }

    fn save_config(&mut self) {
        self.cfg.transform_path = Some(self.transform_path.clone());
        self.cfg.ip             = Some(self.phone_ip.clone());
        self.cfg.source         = Some(if self.source == Source::Webcam { 1 } else { 0 });
        self.cfg.camera_index   = Some(self.selected_cam);
        self.cfg.save();
    }

    fn connect(&mut self) {
        self.active.store(true, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel::<TrackingResponce>();
        let flag  = Arc::clone(&self.active);
        let flag2 = Arc::clone(&self.active);
        let path  = self.transform_path.clone();

        match self.source {
            Source::IPhone => {
                let ip = self.phone_ip.clone();
                thread::spawn(move || VtsPhone::run(ip, tx, flag2));
            }
            Source::Webcam => {
                let idx = self.selected_cam;
                thread::spawn(move || WebcamTracker::run(idx, tx, flag2));
            }
        }
        thread::spawn(move || VtsPc::run(rx, path, flag));
    }

    fn disconnect(&mut self) {
        self.active.store(false, Ordering::Relaxed);
    }

    fn open_file_dialog(&mut self) {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let r = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .pick_file()
                .map(|p| p.to_string_lossy().into_owned());
            let _ = tx.send(r);
        });
        self.pending_path = Some(rx);
    }

    fn can_connect(&self) -> bool {
        !self.transform_path.is_empty()
            && (self.source == Source::Webcam || !self.phone_ip.is_empty())
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll async file dialog
        if let Some(rx) = self.pending_path.take() {
            match rx.try_recv() {
                Ok(Some(path)) => {
                    self.transform_path = path.clone();
                    self.save_config();
                    self.editor.load_file(&path);
                }
                Ok(None) => {}
                Err(_) => { self.pending_path = Some(rx); }
            }
        }

        let connected = self.active.load(Ordering::Relaxed);

        // ── Tab bar (top) ──────────────────────────────────────────────────
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Bridge, "Bridge");
                let cfg_label = if self.editor.dirty { "Config ●" } else { "Config" };
                ui.selectable_value(&mut self.tab, Tab::Config, cfg_label);
            });
            ui.add_space(2.0);
        });

        // ── Status bar (bottom) ────────────────────────────────────────────
        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                let (col, txt) = if connected {
                    (egui::Color32::from_rgb(80, 200, 100), "Connected")
                } else {
                    (egui::Color32::from_gray(110), "Disconnected")
                };
                ui.label(egui::RichText::new("●").color(col).small());
                ui.label(egui::RichText::new(txt).small());

                if !self.transform_path.is_empty() {
                    ui.separator();
                    let fname = PathBuf::from(&self.transform_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    ui.label(egui::RichText::new(fname).small().color(egui::Color32::from_gray(150)));
                }
            });
            ui.add_space(3.0);
        });

        // ── Main content ───────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                Tab::Bridge => bridge_ui(ui, self, connected),
                Tab::Config => config_editor_ui(ui, &mut self.editor, &mut self.transform_path, &mut self.cfg),
            }
        });
    }
}

// ── Bridge tab ────────────────────────────────────────────────────────────────

fn bridge_ui(ui: &mut egui::Ui, app: &mut App, connected: bool) {
    ui.add_space(8.0);

    // Source
    ui.horizontal(|ui| {
        let prev = app.source;
        ui.selectable_value(&mut app.source, Source::IPhone, "iPhone");
        ui.selectable_value(&mut app.source, Source::Webcam, "Webcam");
        if prev != app.source && !connected { app.save_config(); }
    });

    ui.add_space(6.0);

    // Transform path + open button
    ui.horizontal(|ui| {
        let r = ui.add_sized(
            [ui.available_width() - 42.0, 22.0],
            egui::TextEdit::singleline(&mut app.transform_path)
                .hint_text("Transform config (.json)")
                .interactive(!connected),
        );
        if r.changed() { app.save_config(); }
        if ui.add_enabled(!connected,
            egui::Button::new("📂").min_size([36.0, 22.0].into())
        ).on_hover_text("Browse…").clicked() {
            app.open_file_dialog();
        }
    });

    ui.add_space(6.0);

    // Source-specific input
    match app.source {
        Source::IPhone => {
            let r = ui.add_sized(
                [ui.available_width(), 22.0],
                egui::TextEdit::singleline(&mut app.phone_ip)
                    .hint_text("iPhone IP  (e.g. 192.168.1.10)")
                    .interactive(!connected),
            );
            if r.changed() { app.save_config(); }
        }
        Source::Webcam => {
            let snap: Vec<_> = app.cameras.clone();
            let name = snap.iter().find(|(i, _)| *i == app.selected_cam)
                .map(|(_, n)| n.as_str()).unwrap_or("No cameras found");
            let mut new_cam = app.selected_cam;
            egui::ComboBox::from_id_salt("cam")
                .width(ui.available_width())
                .selected_text(name)
                .show_ui(ui, |ui| {
                    for (idx, n) in &snap {
                        ui.selectable_value(&mut new_cam, *idx, n);
                    }
                });
            if new_cam != app.selected_cam {
                app.selected_cam = new_cam;
                app.save_config();
            }
        }
    }

    ui.add_space(12.0);

    // Connect / Disconnect
    let can = app.can_connect();
    let label = if connected { "Disconnect" } else { "Connect" };
    if ui.add_enabled(connected || can,
        egui::Button::new(label).min_size([ui.available_width(), 30.0].into())
    ).clicked() {
        if connected { app.disconnect(); } else { app.connect(); }
    }

    ui.add_space(10.0);

    // Workflow hint
    if !connected {
        let hint = if app.transform_path.is_empty() {
            "① Browse or paste a transform config path"
        } else if app.source == Source::IPhone && app.phone_ip.is_empty() {
            "② Enter your iPhone's IP address"
        } else {
            "② Press Connect — make sure VTube Studio is open"
        };
        ui.label(egui::RichText::new(hint).small().color(egui::Color32::from_gray(140)));
    }

    ui.add_space(6.0);
    ui.separator();
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("github.com/LakoMoor/rusty-bridger")
            .small().color(egui::Color32::from_gray(100)),
    );
}

// ── Config editor tab ─────────────────────────────────────────────────────────

fn config_editor_ui(
    ui: &mut egui::Ui,
    ed: &mut Editor,
    path: &mut String,
    cfg: &mut Config,
) {
    // Toolbar
    ui.horizontal(|ui| {
        if ui.button("📂 Load").clicked() {
            if let Some(p) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
                let s = p.to_string_lossy().into_owned();
                *path = s.clone();
                cfg.transform_path = Some(s.clone());
                cfg.save();
                ed.load_file(&s);
            }
        }
        if ui.button("💾 Save").clicked() {
            if path.is_empty() {
                if let Some(p) = rfd::FileDialog::new().add_filter("JSON", &["json"]).save_file() {
                    *path = p.to_string_lossy().into_owned();
                    cfg.transform_path = Some(path.clone());
                    cfg.save();
                }
            }
            if !path.is_empty() { ed.save_file(path); }
        }
        if ui.button("📋 New").clicked() {
            ed.apply_edit();
            ed.params.clear();
            ed.selected = None;
            ed.dirty = false;
            ed.status = "New config — save when ready".into();
        }

        // Status on the right
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let col = if ed.status.contains("error") || ed.status.contains("Error") {
                egui::Color32::from_rgb(220, 80, 80)
            } else {
                egui::Color32::from_gray(150)
            };
            ui.label(egui::RichText::new(&ed.status).small().color(col));
        });
    });

    ui.add_space(2.0);
    ui.separator();
    ui.add_space(4.0);

    // Split: param list | editor
    let avail = ui.available_height();
    ui.horizontal(|ui| {
        // ── Left: param list ────────────────────────────────────────────
        ui.vertical(|ui| {
            ui.set_width(170.0);

            // List toolbar
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Parameters").strong().small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let has_sel = ed.selected.is_some();
                    let not_last = ed.selected.map_or(false, |i| i + 1 < ed.params.len());
                    let not_first = ed.selected.map_or(false, |i| i > 0);

                    if ui.add_enabled(has_sel && not_last,
                        egui::Button::new("↓").small())
                        .on_hover_text("Move down").clicked() { ed.move_selected(false); }
                    if ui.add_enabled(has_sel && not_first,
                        egui::Button::new("↑").small())
                        .on_hover_text("Move up").clicked() { ed.move_selected(true); }
                    if ui.add_enabled(has_sel,
                        egui::Button::new("🗑").small())
                        .on_hover_text("Delete").clicked() { ed.delete_selected(); }
                    if ui.small_button("＋").on_hover_text("Add param").clicked() {
                        ed.add_param();
                    }
                });
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(avail - 36.0)
                .show(ui, |ui| {
                    ui.set_width(160.0);
                    for i in 0..ed.params.len() {
                        let sel = ed.selected == Some(i);
                        let p = &ed.params[i];
                        let color = if sel {
                            egui::Color32::WHITE
                        } else {
                            egui::Color32::from_gray(200)
                        };
                        let label = egui::RichText::new(&p.name).monospace().small().color(color);
                        if ui.selectable_label(sel, label).clicked() && !sel {
                            ed.apply_edit();
                            ed.select(i);
                        }
                    }
                    if ed.params.is_empty() {
                        ui.label(
                            egui::RichText::new("No params.\nPress ＋ to add one.")
                                .small().color(egui::Color32::from_gray(130)),
                        );
                    }
                });
        });

        ui.separator();

        // ── Right: param editor ─────────────────────────────────────────
        ui.vertical(|ui| {
            if ed.selected.is_none() {
                ui.add_space(60.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("← Select a parameter to edit")
                            .color(egui::Color32::from_gray(140)),
                    );
                });
                return;
            }

            let w = ui.available_width();

            egui::Grid::new("param_grid")
                .num_columns(2)
                .spacing([8.0, 5.0])
                .min_col_width(56.0)
                .show(ui, |ui| {
                    // Name
                    ui.label("Name");
                    ui.horizontal(|ui| {
                        let r = ui.add_sized(
                            [w - 80.0, 22.0],
                            egui::TextEdit::singleline(&mut ed.buf_name),
                        );
                        if r.changed() { ed.apply_edit(); ed.validate_buffers(); }
                        if ed.name_dup {
                            ui.label(
                                egui::RichText::new("⚠ dup").small()
                                    .color(egui::Color32::from_rgb(240, 160, 30)),
                            );
                        }
                    });
                    ui.end_row();

                    // Formula
                    ui.label("Formula");
                    ui.horizontal(|ui| {
                        let r = ui.add_sized(
                            [w - 80.0, 22.0],
                            egui::TextEdit::singleline(&mut ed.buf_func)
                                .font(egui::TextStyle::Monospace)
                                .hint_text("HeadRotY * -1"),
                        );
                        if r.changed() { ed.apply_edit(); ed.validate_buffers(); }
                        match ed.formula_ok {
                            Some(true)  => { ui.label(egui::RichText::new("✓").color(egui::Color32::from_rgb(80, 200, 100))); }
                            Some(false) => { ui.label(egui::RichText::new("✗").color(egui::Color32::from_rgb(220, 80, 80))); }
                            None => {}
                        }
                    });
                    ui.end_row();

                    // Min / Max / Default on one row
                    ui.label("Range");
                    ui.horizontal(|ui| {
                        let fw = (w - 100.0) / 3.0;
                        ui.label(egui::RichText::new("min").small().color(egui::Color32::from_gray(160)));
                        let r = ui.add_sized([fw, 20.0], egui::TextEdit::singleline(&mut ed.buf_min));
                        if r.changed() { ed.apply_edit(); ed.validate_buffers(); }
                        ui.label(egui::RichText::new("max").small().color(egui::Color32::from_gray(160)));
                        let r = ui.add_sized([fw, 20.0], egui::TextEdit::singleline(&mut ed.buf_max));
                        if r.changed() { ed.apply_edit(); ed.validate_buffers(); }
                        ui.label(egui::RichText::new("def").small().color(egui::Color32::from_gray(160)));
                        let r = ui.add_sized([fw, 20.0], egui::TextEdit::singleline(&mut ed.buf_default));
                        if r.changed() { ed.apply_edit(); ed.validate_buffers(); }
                    });
                    ui.end_row();
                });

            ui.add_space(6.0);

            // Range bar
            let min_v: f64 = ed.buf_min.parse().unwrap_or(-1.0);
            let max_v: f64 = ed.buf_max.parse().unwrap_or(1.0);
            let def_v: f64 = ed.buf_default.parse().unwrap_or(0.0);
            let t = if (max_v - min_v).abs() > 1e-9 {
                ((def_v - min_v) / (max_v - min_v)).clamp(0.0, 1.0) as f32
            } else { 0.5 };

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.2}", min_v)).small().monospace().color(egui::Color32::from_gray(160)));
                let bar_w = ui.available_width() - 44.0;
                let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, 8.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 3.0, egui::Color32::from_gray(50));
                let fill = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * t, rect.height()));
                ui.painter().rect_filled(fill, 3.0, egui::Color32::from_rgb(80, 160, 240));
                // default tick
                let tick_x = rect.min.x + rect.width() * t;
                ui.painter().line_segment(
                    [egui::pos2(tick_x, rect.min.y - 2.0), egui::pos2(tick_x, rect.max.y + 2.0)],
                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                );
                ui.label(egui::RichText::new(format!("{:.2}", max_v)).small().monospace().color(egui::Color32::from_gray(160)));
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // Available variables (collapsible)
            egui::CollapsingHeader::new(
                egui::RichText::new("Available variables").small()
            )
            .default_open(false)
            .show(ui, |ui| {
                let vars = [
                    ("Head rotation",   "HeadRotX  HeadRotY  HeadRotZ"),
                    ("Head position",   "HeadPosX  HeadPosY  HeadPosZ"),
                    ("Eyes",            "EyeBlinkLeft  EyeBlinkRight"),
                    ("Mouth",           "JawOpen  MouthSmileLeft  MouthSmileRight\nMouthFrownLeft  MouthFrownRight"),
                    ("Brows",           "BrowOuterUpLeft  BrowOuterUpRight"),
                ];
                egui::Grid::new("vars_grid").num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                    for (cat, names) in &vars {
                        ui.label(egui::RichText::new(*cat).small().color(egui::Color32::from_gray(140)));
                        ui.label(egui::RichText::new(*names).small().monospace().color(egui::Color32::from_gray(200)));
                        ui.end_row();
                    }
                });
                ui.add_space(2.0);
                ui.label(egui::RichText::new("Operators: + - * / ^ ( ) math functions").small().color(egui::Color32::from_gray(130)));
            });
        });
    });
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let data_dir = app_dir();
    let _ = std::env::set_current_dir(&data_dir);

    let log_dir  = data_dir.join("log");
    let _ = fs::create_dir_all(&log_dir);
    let log_file = log_dir.join("log.log").to_string_lossy().into_owned();
    let log_cfg  = include_str!("../../configs/log_cfg.yml")
        .replace("log/log.log", &log_file);
    if let Ok(raw) = serde_yaml::from_str(&log_cfg) {
        let _ = log4rs::init_raw_config(raw);
    }

    rusty_bridge_lib::webcam::init_camera_permissions();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rusty Bridge")
            .with_inner_size([420.0, 480.0])
            .with_min_inner_size([360.0, 380.0])
            .with_resizable(true),
        ..Default::default()
    };

    eframe::run_native(
        "Rusty Bridge",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .unwrap();
}
