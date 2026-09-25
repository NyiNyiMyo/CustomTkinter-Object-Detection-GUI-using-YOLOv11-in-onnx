import ast
import random
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
except ImportError:
    print("ERROR: customtkinter not installed!")
    print("Install with: pip install customtkinter")
    exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    print("(For GPU acceleration: pip install onnxruntime-gpu)")
    exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Dark mode color scheme
COLORS = {
    'bg': '#1e1e1e',
    'bg_light': '#2d2d2d',
    'bg_lighter': '#3d3d3d',
    'fg': '#ffffff',
    'fg_dim': '#b0b0b0',
    'accent': '#007acc',
    'accent_hover': '#005a9e',
    'success': '#4ec9b0',
    'warning': '#ce9178',
    'error': '#f48771',
    'border': '#404040',
    'danger': '#c0392b',
    'danger_hover': '#992d22',
    'green': '#2ecc71',
    'green_hover': '#27ae60',
}

# A fixed color palette (BGR) used to draw boxes per class id
BOX_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (192, 192, 192), (64, 64, 255), (64, 255, 64), (255, 64, 64), (200, 200, 0),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
PLACEHOLDER = "📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start"

# Grid settings for folder inference (each cell is rendered at this size)
GRID_COLS, GRID_ROWS = 2, 2
GRID_COUNT = GRID_COLS * GRID_ROWS
CELL_W, CELL_H = 640, 480


def F(size=12, bold=False, family='Arial'):
    return ctk.CTkFont(family=family, size=size, weight='bold' if bold else 'normal')


# ============================================================
# ONNX inference (unchanged logic)
# ============================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns the padded image, the resize ratio, and (dw, dh) half-padding."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class SimpleBox:
    """Mimics the small subset of ultralytics' Boxes API the UI code relies on."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class SimpleResult:
    """Mimics the small subset of ultralytics' Results API the UI code relies on."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class ONNXYOLO:
    """Thin ONNX Runtime wrapper that reproduces the predict() -> result interface
    the rest of the app expects, so the GUI code barely has to change."""

    def __init__(self, model_path):
        providers = []
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        self.names = self._load_names()

    def _load_names(self):
        names = {}
        try:
            meta = self.session.get_modelmeta()
            custom = meta.custom_metadata_map
            if custom and 'names' in custom:
                parsed = ast.literal_eval(custom['names'])
                if isinstance(parsed, dict):
                    names = {int(k): v for k, v in parsed.items()}
                elif isinstance(parsed, (list, tuple)):
                    names = {i: v for i, v in enumerate(parsed)}
        except Exception:
            names = {}

        if not names:
            names = {i: f"class{i}" for i in range(1000)}
        return names

    def _postprocess(self, pred, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
        orig_h, orig_w = orig_shape[:2]

        pred = pred[0]
        # Normalize to shape (num_anchors, 4 + num_classes)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return [], [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores > conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return [], [], []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox padding/scale to map back to original image coordinates
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thres, iou_thres)
        if indices is None or len(indices) == 0:
            return [], [], []
        indices = np.array(indices).flatten()

        final_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]
        final_scores = [float(scores[i]) for i in indices]
        final_class_ids = [int(class_ids[i]) for i in indices]

        return final_boxes, final_scores, final_class_ids

    def _draw(self, img_bgr, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = box.cls[0]
            conf = box.conf[0]
            name = self.names.get(cls_id, f"class{cls_id}")
            color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    def infer(self, image_bgr, conf=0.25, iou=0.45):
        """Run detection on a BGR numpy image. Returns (SimpleResult, annotated_bgr)."""
        letter_img, ratio, (dw, dh) = letterbox(image_bgr, (self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        boxes, scores, class_ids = self._postprocess(pred, conf, iou, ratio, dw, dh, image_bgr.shape)
        simple_boxes = [SimpleBox(class_ids[i], scores[i], boxes[i]) for i in range(len(boxes))]
        result = SimpleResult(simple_boxes, self.names)

        annotated = self._draw(image_bgr.copy(), simple_boxes)
        return result, annotated


def fit_into_cell(img_bgr, w, h, bg=(45, 45, 45)):
    """Fit an image into a w x h cell (keep aspect ratio, centered)."""
    ih, iw = img_bgr.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((h, w, 3), bg, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    return cell


def build_grid(annotated_list, names_list):
    """Compose annotated BGR images into a GRID_COLS x GRID_ROWS grid (BGR)."""
    cells = []
    for img, name in zip(annotated_list, names_list):
        cell = fit_into_cell(img, CELL_W, CELL_H)
        label = name if len(name) <= 40 else name[:37] + "..."
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(cell, (0, CELL_H - th - bl - 10), (tw + 12, CELL_H), (30, 30, 30), -1)
        cv2.putText(cell, label, (6, CELL_H - bl - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), (64, 64, 64), 2)
        cells.append(cell)
    while len(cells) < GRID_COUNT:
        cells.append(np.full((CELL_H, CELL_W, 3), (45, 45, 45), dtype=np.uint8))
    rows = [np.hstack(cells[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


class Page:
    """Holds the widgets/state that belong to one mode page."""

    def __init__(self, mode):
        self.mode = mode
        self.frame = None
        self.canvas = None
        self.canvas_text = None
        self.placeholder_visible = True
        self.photo = None
        self.file_label = None
        self.select_btn = None
        self.stop_btn = None
        self.clear_btn = None
        self.save_btn = None
        self.swap_btn = None
        self.shuffle_btn = None
        self.pause_btn = None
        self.progress = None
        self.result_frame = None
        self.last_result = None


class YOLODetectorApp(ctk.CTk):
    """
    YOLO Object Detection - CustomTkinter GUI Application (Dark Mode)
    One sidebar, one page per mode (Image / Video / Webcam / Folder),
    ONNX model via ONNX Runtime.
    """

    MODES = [
        ('image', "🖼 Image"),
        ('video', "🎬 Video"),
        ('webcam', "📹 Webcam"),
        ('folder', "📂 Folder"),
    ]

    def __init__(self):
        super().__init__()
        self.title("YOLO Object Detection - Dark Mode (ONNX)")
        self.geometry("1400x850")
        self.minsize(1100, 700)
        self.configure(fg_color=COLORS['bg'])
        self.after(100, self.maximize_window)

        # Variables
        self.model = None
        self.model_path = None
        self.current_file = None
        self.folder_path = None
        self.video_thread = None
        self.stop_video = False
        self.is_processing = False
        self.active_page = None
        self._ui_busy = False
        self.paused = False
        self.conf_value = 0.30
        self.iou_value = 0.30

        # Camera state
        self.available_cameras = [0]
        self.current_cam_index = 0
        self._swapping = False

        self.pages = {}
        self.nav_buttons = {}

        self.setup_ui()
        self.show_page('image')

        # Auto-load model from models folder
        self.after(200, self.auto_load_model)

        # Detect cameras in the background so we don't block startup
        threading.Thread(target=self._detect_cameras_bg, daemon=True).start()

    def maximize_window(self):
        """Maximize the window on launch, across platforms."""
        try:
            self.state('zoomed')  # Windows / some Linux window managers
        except tk.TclError:
            try:
                self.attributes('-zoomed', True)  # Most Linux WMs
            except tk.TclError:
                w = self.winfo_screenwidth()
                h = self.winfo_screenheight()
                self.geometry(f"{w}x{h}+0+0")

    # ============================================================
    # UI construction
    # ============================================================
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()

        main = ctk.CTkFrame(self, fg_color=COLORS['bg'], corner_radius=0)
        main.grid(row=0, column=1, sticky='nsew')
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.page_container = ctk.CTkFrame(main, fg_color=COLORS['bg'], corner_radius=0)
        self.page_container.grid(row=0, column=0, sticky='nsew', padx=15, pady=(15, 8))
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        for mode, _ in self.MODES:
            self.pages[mode] = self._build_page(mode)

        # Status bar
        self.status_label = ctk.CTkLabel(
            main, text="Ready", anchor='w', height=34, corner_radius=8,
            fg_color=COLORS['bg_light'], text_color=COLORS['fg'], font=F(12))
        self.status_label.grid(row=1, column=0, sticky='ew', padx=15, pady=(0, 15))

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=270, corner_radius=0,
                                         fg_color=COLORS['bg_light'])
        sidebar.grid(row=0, column=0, sticky='nsw')
        self.sidebar = sidebar

        ctk.CTkLabel(sidebar, text="📟 Makers", font=F(22, True),
                     text_color=COLORS['accent']).pack(anchor='w', padx=12, pady=(12, 0))
        ctk.CTkLabel(sidebar, text="YOLO Object Detection", font=F(12),
                     text_color=COLORS['fg_dim']).pack(anchor='w', padx=12, pady=(0, 14))

        # --- Model ---
        ctk.CTkLabel(sidebar, text="🤖 ONNX Model", font=F(13, True),
                     text_color=COLORS['accent']).pack(anchor='w', padx=12, pady=(4, 4))
        model_card = ctk.CTkFrame(sidebar, fg_color=COLORS['bg_lighter'], corner_radius=10)
        model_card.pack(fill='x', padx=8, pady=(0, 12))

        ctk.CTkLabel(model_card, text="Current Model:", font=F(11),
                     text_color=COLORS['fg_dim']).pack(anchor='w', padx=12, pady=(10, 0))
        self.model_label = ctk.CTkLabel(model_card, text="Loading...", font=F(12, True),
                                        text_color=COLORS['error'], wraplength=210,
                                        justify='left')
        self.model_label.pack(anchor='w', padx=12, pady=(0, 8))

        ctk.CTkButton(model_card, text="📂 Change Model", command=self.load_model,
                      fg_color=COLORS['bg_light'], hover_color=COLORS['accent_hover'],
                      font=F(12)).pack(fill='x', padx=12, pady=(0, 6))
        ctk.CTkButton(model_card, text="🔄 Reload", command=self.reload_model,
                      fg_color=COLORS['bg_light'], hover_color=COLORS['accent_hover'],
                      font=F(12)).pack(fill='x', padx=12, pady=(0, 12))

        # --- Mode navigation ---
        ctk.CTkLabel(sidebar, text="📁 Select Option", font=F(13, True),
                     text_color=COLORS['accent']).pack(anchor='w', padx=12, pady=(4, 4))
        for mode, text in self.MODES:
            btn = ctk.CTkButton(sidebar, text=text, anchor='w', height=40, font=F(13, True),
                                fg_color='transparent', hover_color=COLORS['bg_lighter'],
                                text_color=COLORS['fg'],
                                command=lambda m=mode: self.show_page(m))
            btn.pack(fill='x', padx=8, pady=2)
            self.nav_buttons[mode] = btn

        # --- Settings ---
        ctk.CTkLabel(sidebar, text="⚙️ Detection Settings", font=F(13, True),
                     text_color=COLORS['accent']).pack(anchor='w', padx=12, pady=(18, 4))
        settings_card = ctk.CTkFrame(sidebar, fg_color=COLORS['bg_lighter'], corner_radius=10)
        settings_card.pack(fill='x', padx=8, pady=(0, 12))

        # Confidence
        row = ctk.CTkFrame(settings_card, fg_color='transparent')
        row.pack(fill='x', padx=12, pady=(10, 0))
        ctk.CTkLabel(row, text="Confidence Threshold:", font=F(11)).pack(side='left')
        self.conf_label = ctk.CTkLabel(row, text="0.30", font=F(12, True),
                                       text_color=COLORS['accent'])
        self.conf_label.pack(side='right')
        self.conf_var = tk.DoubleVar(value=0.30)
        self.conf_scale = ctk.CTkSlider(settings_card, from_=0.1, to=0.9, number_of_steps=80,
                                        variable=self.conf_var, command=self.update_conf_label,
                                        progress_color=COLORS['accent'],
                                        button_color=COLORS['accent'],
                                        button_hover_color=COLORS['accent_hover'])
        self.conf_scale.pack(fill='x', padx=12, pady=(4, 10))

        # IoU
        row = ctk.CTkFrame(settings_card, fg_color='transparent')
        row.pack(fill='x', padx=12)
        ctk.CTkLabel(row, text="IoU Threshold:", font=F(11)).pack(side='left')
        self.iou_label = ctk.CTkLabel(row, text="0.30", font=F(12, True),
                                      text_color=COLORS['accent'])
        self.iou_label.pack(side='right')
        self.iou_var = tk.DoubleVar(value=0.30)
        self.iou_scale = ctk.CTkSlider(settings_card, from_=0.1, to=0.9, number_of_steps=80,
                                       variable=self.iou_var, command=self.update_iou_label,
                                       progress_color=COLORS['accent'],
                                       button_color=COLORS['accent'],
                                       button_hover_color=COLORS['accent_hover'])
        self.iou_scale.pack(fill='x', padx=12, pady=(4, 12))

    def _build_page(self, mode):
        page = Page(mode)
        page.frame = ctk.CTkFrame(self.page_container, fg_color=COLORS['bg'], corner_radius=0)
        page.frame.grid_columnconfigure(0, weight=1)
        page.frame.grid_rowconfigure(2, weight=1)

        title = dict(self.MODES)[mode]

        # Header: title + selected file
        header = ctk.CTkFrame(page.frame, fg_color=COLORS['bg_light'], corner_radius=10)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ctk.CTkLabel(header, text=title, font=F(18, True),
                     text_color=COLORS['fg']).pack(side='left', padx=(15, 20), pady=10)
        ctk.CTkLabel(header, text="📄 Selected File:", font=F(11),
                     text_color=COLORS['fg_dim']).pack(side='left', padx=(0, 5))
        page.file_label = ctk.CTkLabel(header, text="No file selected", font=F(12, True),
                                       text_color=COLORS['error'])
        page.file_label.pack(side='left', padx=5)

        # Toolbar: mode-specific action + common actions
        toolbar = ctk.CTkFrame(page.frame, fg_color='transparent')
        toolbar.grid(row=1, column=0, sticky='ew', pady=(0, 10))

        select_cmd = {
            'image': self.upload_image,
            'video': self.upload_video,
            'webcam': self.use_webcam,
            'folder': self.upload_folder,
        }[mode]
        page.select_btn = ctk.CTkButton(
            toolbar, text=title, width=150, height=36, font=F(13, True),
            fg_color=COLORS['accent'], hover_color=COLORS['accent_hover'],
            command=select_cmd)
        page.select_btn.pack(side='left', padx=(0, 8))

        if mode == 'folder':
            page.shuffle_btn = ctk.CTkButton(
                toolbar, text="🎲 Shuffle 4 Images", width=170, height=36, font=F(13, True),
                fg_color=COLORS['accent'], hover_color=COLORS['accent_hover'],
                state='disabled', command=self.shuffle_folder)
            page.shuffle_btn.pack(side='left', padx=(0, 8))

        if mode == 'webcam':
            page.swap_btn = ctk.CTkButton(
                toolbar, text="🔀 Swap Cam", width=130, height=36, font=F(13, True),
                fg_color=COLORS['accent'], hover_color=COLORS['accent_hover'],
                state='disabled', command=self.swap_camera)
            page.swap_btn.pack(side='left', padx=(0, 8))

        if mode == 'video':
            page.pause_btn = ctk.CTkButton(
                toolbar, text="⏸ Pause", width=110, height=36, font=F(13, True),
                fg_color=COLORS['accent'], hover_color=COLORS['accent_hover'],
                state='disabled', command=self.toggle_pause)
            page.pause_btn.pack(side='left', padx=(0, 8))

        if mode in ('video', 'webcam'):
            page.stop_btn = ctk.CTkButton(
                toolbar, text="⏹ Stop", width=100, height=36, font=F(13, True),
                fg_color=COLORS['danger'], hover_color=COLORS['danger_hover'],
                state='disabled', command=self.stop_detection)
            page.stop_btn.pack(side='left', padx=(0, 8))

        page.clear_btn = ctk.CTkButton(
            toolbar, text="🗑 Clear", width=100, height=36, font=F(13, True),
            fg_color=COLORS['danger'], hover_color=COLORS['danger_hover'],
            command=self.clear_display)
        page.clear_btn.pack(side='left', padx=(0, 8))

        if mode == 'webcam':
            page.save_btn = ctk.CTkButton(
                toolbar, text="📸 Capture", width=140, height=36, font=F(13, True),
                fg_color=COLORS['green'], hover_color=COLORS['green_hover'],
                text_color='#0b2e1a', state='disabled', command=self.capture_frame)
            page.save_btn.pack(side='left', padx=(0, 8))
        elif mode != 'video':
            page.save_btn = ctk.CTkButton(
                toolbar, text="💾 Save Result", width=140, height=36, font=F(13, True),
                fg_color=COLORS['green'], hover_color=COLORS['green_hover'],
                text_color='#0b2e1a', state='disabled', command=self.save_result)
            page.save_btn.pack(side='left', padx=(0, 8))

        # Content: display + results
        content = ctk.CTkFrame(page.frame, fg_color='transparent')
        content.grid(row=2, column=0, sticky='nsew')
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        display_card = ctk.CTkFrame(content, fg_color=COLORS['bg_light'], corner_radius=12)
        display_card.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        display_card.grid_columnconfigure(0, weight=1)
        display_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(display_card, text="📺 Display", font=F(13, True),
                     text_color=COLORS['accent']).grid(row=0, column=0, sticky='w',
                                                       padx=14, pady=(10, 4))

        page.canvas = tk.Canvas(display_card, bg=COLORS['bg'], highlightthickness=0, bd=0)
        page.canvas.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 6))
        page.canvas_text = page.canvas.create_text(
            400, 300, text=PLACEHOLDER, fill=COLORS['fg_dim'],
            font=('Arial', 16), justify=tk.CENTER)
        page.canvas.bind('<Configure>', lambda e, p=page: self.on_canvas_resize(p, e))

        # Progress bar - only shown while actively processing
        page.progress = ctk.CTkProgressBar(display_card, mode='indeterminate',
                                           progress_color=COLORS['accent'],
                                           fg_color=COLORS['bg_lighter'])

        results_card = ctk.CTkFrame(content, fg_color=COLORS['bg_light'], corner_radius=12,
                                    width=390)
        results_card.grid(row=0, column=1, sticky='ns', padx=(8, 0))
        results_card.grid_propagate(False)
        results_card.grid_columnconfigure(0, weight=1)
        results_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(results_card, text="📊 Detection Results", font=F(13, True),
                     text_color=COLORS['accent']).grid(row=0, column=0, sticky='w',
                                                       padx=14, pady=(10, 4))
        page.result_frame = ctk.CTkScrollableFrame(results_card, fg_color=COLORS['bg'],
                                                   corner_radius=8)
        page.result_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))

        self.show_no_results(page)
        return page

    # ============================================================
    # Page switching
    # ============================================================
    def show_page(self, mode):
        # Switching pages while a stream is running: stop it first
        if self.is_processing and self.active_page and self.active_page.mode != mode:
            self.stop_video = True

        for m, page in self.pages.items():
            page.frame.grid_forget()
        self.active_page = self.pages[mode]
        self.active_page.frame.grid(row=0, column=0, sticky='nsew')

        for m, btn in self.nav_buttons.items():
            if m == mode:
                btn.configure(fg_color=COLORS['accent'], hover_color=COLORS['accent_hover'])
            else:
                btn.configure(fg_color='transparent', hover_color=COLORS['bg_lighter'])

    # ============================================================
    # Small helpers
    # ============================================================
    def ui(self, fn):
        """Schedule fn on the UI thread (safe to call from worker threads)."""
        try:
            self.after(0, fn)
        except Exception:
            pass

    def on_canvas_resize(self, page, event):
        """Keep the placeholder text perfectly centered in the display canvas."""
        if page.placeholder_visible:
            try:
                page.canvas.coords(page.canvas_text, event.width // 2, event.height // 2)
            except tk.TclError:
                pass

    def update_conf_label(self, value=None):
        self.conf_value = float(self.conf_var.get())
        self.conf_label.configure(text=f"{self.conf_value:.2f}")

    def update_iou_label(self, value=None):
        self.iou_value = float(self.iou_var.get())
        self.iou_label.configure(text=f"{self.iou_value:.2f}")

    def update_status(self, message, color=None):
        self.status_label.configure(text=message,
                                    text_color=color if color else COLORS['fg'])

    def draw_rgb(self, page, rgb, fill=False):
        """Fit an RGB numpy image into the page canvas.
        fill=True also scales small frames UP (used for video/webcam, whose
        native resolution is often much smaller than the display area)."""
        page.placeholder_visible = False
        img = Image.fromarray(rgb)
        cw = page.canvas.winfo_width()
        ch = page.canvas.winfo_height()
        if cw <= 1:
            cw = 800
        if ch <= 1:
            ch = 600
        if fill:
            r = min((cw - 20) / img.width, (ch - 20) / img.height)
            new_size = (max(1, int(img.width * r)), max(1, int(img.height * r)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        else:
            img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)
        page.photo = ImageTk.PhotoImage(img)
        page.canvas.delete("all")
        page.canvas.create_image(cw // 2, ch // 2, image=page.photo, anchor=tk.CENTER)

    def _require_model(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return False
        return True

    def _begin_job(self, page, show_progress):
        self.is_processing = True
        self.stop_video = False
        page.select_btn.configure(state='disabled')
        if page.shuffle_btn:
            page.shuffle_btn.configure(state='disabled')
        if page.stop_btn:
            page.stop_btn.configure(state='normal')
        if page.pause_btn:
            self.paused = False
            page.pause_btn.configure(state='normal', text="⏸ Pause")
        if page.mode == 'webcam':
            page.save_btn.configure(state='normal')
        if show_progress:
            page.progress.grid(row=2, column=0, sticky='ew', padx=14, pady=(0, 10))
            page.progress.start()

    def _end_job(self, page):
        try:
            page.progress.stop()
            page.progress.grid_forget()
        except Exception:
            pass
        page.select_btn.configure(state='normal')
        if page.shuffle_btn and self.folder_path:
            page.shuffle_btn.configure(state='normal')
        if page.stop_btn:
            page.stop_btn.configure(state='disabled')
        if page.pause_btn:
            self.paused = False
            page.pause_btn.configure(state='disabled', text="⏸ Pause")
        self.is_processing = False

    # ============================================================
    # Camera detection / swapping
    # ============================================================
    def _detect_cameras_bg(self):
        """Probe a few camera indices in a background thread."""
        found = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                cap.release()
            except Exception:
                pass
        if not found:
            found = [0]
        self.ui(lambda: self._on_cameras_detected(found))

    def _on_cameras_detected(self, cams):
        self.available_cameras = cams
        self.current_cam_index = cams[0]
        btn = self.pages['webcam'].swap_btn
        btn.configure(state=tk.NORMAL if len(cams) > 1 else tk.DISABLED)

    def swap_camera(self):
        """Cycle to the next detected camera index.

        Clean stop -> wait for the capture to fully release -> restart, so the
        old and new camera handles never overlap."""
        if len(self.available_cameras) < 2:
            return
        if self._swapping:
            return

        page = self.pages['webcam']
        try:
            idx = self.available_cameras.index(self.current_cam_index)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(self.available_cameras)
        new_index = self.available_cameras[idx]
        self.current_cam_index = new_index

        if not isinstance(self.current_file, int):
            self.update_status(
                f"Camera set to index {new_index} (used next time you select Webcam)",
                COLORS['accent'])
            return

        self._swapping = True
        page.swap_btn.configure(state=tk.DISABLED)

        if self.is_processing:
            self.update_status(f"Switching to camera {new_index}...", COLORS['warning'])
            self.stop_video = True
            old_thread = self.video_thread
            threading.Thread(target=self._wait_and_restart_camera,
                             args=(old_thread, new_index), daemon=True).start()
        else:
            self.current_file = new_index
            page.file_label.configure(text=f"📹 Webcam {new_index}",
                                      text_color=COLORS['success'])
            self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
            self._swapping = False
            page.swap_btn.configure(state=tk.NORMAL)

    def _wait_and_restart_camera(self, old_thread, new_index):
        if old_thread is not None:
            old_thread.join(timeout=5)
        self.ui(lambda: self._finish_camera_swap(new_index))

    def _finish_camera_swap(self, new_index):
        page = self.pages['webcam']
        self.current_file = new_index
        page.file_label.configure(text=f"📹 Webcam {new_index}", text_color=COLORS['success'])
        self._swapping = False
        page.swap_btn.configure(state=tk.NORMAL if len(self.available_cameras) > 1 else tk.DISABLED)
        self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
        # Seamlessly resume detection on the new camera
        self.run_video(page, self.current_file)

    # ============================================================
    # Model loading (ONNX)
    # ============================================================
    def auto_load_model(self):
        """Auto-load model from models folder"""
        models_folder = Path('models')
        model_path = models_folder / 'yolo11n.onnx'

        if model_path.exists():
            self.update_status("Loading default model from models/yolo11n.onnx...", COLORS['warning'])
            self.load_model_file(str(model_path))
        else:
            self.update_status("No model found in 'models' folder. Please select a model.", COLORS['error'])
            self.model_label.configure(text="No model loaded", text_color=COLORS['error'])

            if messagebox.askyesno("Model Not Found",
                                   "Model file 'models/yolo11n.onnx' not found.\n\n"
                                   "Would you like to select an ONNX model file now?"):
                self.load_model()

    def load_model(self):
        """Load YOLO ONNX model from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO ONNX Model File",
            filetypes=[("ONNX Models", "*.onnx"), ("All files", "*.*")],
            initialdir="models" if Path("models").exists() else "."
        )
        if not file_path:
            return
        self.load_model_file(file_path)

    def load_model_file(self, file_path):
        """Load ONNX model from given path"""
        try:
            self.update_status("Loading model...", COLORS['warning'])
            self.update_idletasks()
            self.model = ONNXYOLO(file_path)
            self.model_path = file_path

            model_name = Path(file_path).name
            self.model_label.configure(text=f"✓ {model_name}", text_color=COLORS['success'])
            self.update_status(f"Model loaded successfully: {model_name}", COLORS['success'])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_label.configure(text="❌ Failed to load", text_color=COLORS['error'])
            self.update_status("Error loading model", COLORS['error'])

    def reload_model(self):
        """Reload the current model"""
        if self.model_path:
            self.load_model_file(self.model_path)
        else:
            messagebox.showinfo("Info", "No model to reload. Please load a model first.")

    # ============================================================
    # Selection actions - each one starts detection immediately
    # ============================================================
    def upload_image(self):
        """Select an image and run detection immediately"""
        if not self._require_model():
            return
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        page = self.pages['image']
        self.current_file = file_path
        page.file_label.configure(text=Path(file_path).name, text_color=COLORS['success'])
        self.update_status(f"Image loaded: {Path(file_path).name}", COLORS['success'])
        self.run_image(page, file_path)

    def upload_video(self):
        """Select a video and run detection immediately"""
        if not self._require_model():
            return
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        page = self.pages['video']
        self.current_file = file_path
        page.file_label.configure(text=Path(file_path).name, text_color=COLORS['success'])
        self.update_status(f"Video loaded: {Path(file_path).name}", COLORS['success'])
        self.run_video(page, file_path)

    def use_webcam(self):
        """Use webcam for detection - starts immediately"""
        if not self._require_model():
            return
        page = self.pages['webcam']
        self.current_file = self.current_cam_index
        page.file_label.configure(text=f"📹 Webcam {self.current_cam_index}",
                                  text_color=COLORS['success'])
        self.update_status(f"Webcam {self.current_cam_index} selected", COLORS['success'])
        self.run_video(page, self.current_file)

    def upload_folder(self):
        """Select a folder, randomly pick 4 images and run batch detection"""
        if not self._require_model():
            return
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return

        images = [p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            messagebox.showwarning("Warning", "No images found in the selected folder!")
            return

        page = self.pages['folder']
        self.folder_path = folder
        self.folder_images = images
        page.file_label.configure(text=f"{Path(folder).name}  ({len(images)} images)",
                                  text_color=COLORS['success'])
        self.update_status(f"Folder loaded: {Path(folder).name} ({len(images)} images)",
                           COLORS['success'])
        self.shuffle_folder()

    def shuffle_folder(self):
        """Pick a new random set of up to 4 images and run detection"""
        if not self._require_model() or not getattr(self, 'folder_images', None):
            return
        page = self.pages['folder']
        picks = random.sample(self.folder_images, min(GRID_COUNT, len(self.folder_images)))
        self.run_folder(page, picks)

    # ============================================================
    # Image inference
    # ============================================================
    def run_image(self, page, path):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status("Processing image...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._image_worker, args=(page, path, conf, iou),
                         daemon=True).start()

    def _image_worker(self, page, path, conf, iou):
        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError("Could not read the selected image file")
            result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._image_done(page, result, annotated_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Detection failed", msg))

    def _image_done(self, page, result, annotated_rgb):
        self.draw_rgb(page, annotated_rgb)
        self.display_results(page, result)
        page.last_result = annotated_rgb
        page.save_btn.configure(state=tk.NORMAL)
        self.update_status(f"✓ Detection complete: {len(result.boxes)} objects found",
                           COLORS['success'])
        self._end_job(page)

    def _job_failed(self, page, status, detail):
        messagebox.showerror("Error", f"{status}:\n{detail}")
        self.update_status(status, COLORS['error'])
        self._end_job(page)

    # ============================================================
    # Folder (batch) inference -> 2x2 grid
    # ============================================================
    def run_folder(self, page, paths):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status(f"Processing {len(paths)} images...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._folder_worker, args=(page, paths, conf, iou),
                         daemon=True).start()

    def _folder_worker(self, page, paths, conf, iou):
        try:
            entries, annotated_list, names_list = [], [], []
            for p in paths:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
                entries.append((p.name, result))
                annotated_list.append(annotated_bgr)
                names_list.append(p.name)
            if not entries:
                raise ValueError("None of the selected images could be read")

            grid_bgr = build_grid(annotated_list, names_list)
            grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._folder_done(page, entries, grid_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Batch detection failed", msg))

    def _folder_done(self, page, entries, grid_rgb):
        self.draw_rgb(page, grid_rgb)
        self.display_folder_results(page, entries)
        page.last_result = grid_rgb
        page.save_btn.configure(state=tk.NORMAL)
        total = sum(len(r.boxes) for _, r in entries)
        self.update_status(
            f"✓ Detection complete: {total} objects found in {len(entries)} images",
            COLORS['success'])
        self._end_job(page)

    # ============================================================
    # Video / webcam inference
    # ============================================================
    def run_video(self, page, source):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=False)
        self.video_thread = threading.Thread(target=self._video_worker,
                                             args=(page, source), daemon=True)
        self.video_thread.start()

    def _push_frame(self, page, rgb, text):
        """Drop frames if the UI hasn't finished drawing the previous one."""
        if self._ui_busy:
            return
        self._ui_busy = True
        self.ui(lambda: self._show_frame(page, rgb, text))

    def _show_frame(self, page, rgb, text):
        try:
            self.draw_rgb(page, rgb, fill=True)
            page.last_result = rgb
            self.update_status(text, COLORS['warning'])
        finally:
            self._ui_busy = False

    def _video_worker(self, page, source):
        frame_count = 0
        total_detections = 0
        error = None
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.ui(lambda: messagebox.showerror("Error", "Failed to open video source!"))
                cap.release()
                self.ui(lambda: self._video_done(page, 0, 0, True, None))
                return

            while cap.isOpened() and not self.stop_video:
                if self.paused:
                    time.sleep(0.05)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result, annotated_bgr = self.model.infer(
                    frame, conf=self.conf_value, iou=self.iou_value)
                total_detections += len(result.boxes)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                self._push_frame(
                    page, annotated_rgb,
                    f"Frame {frame_count} - {len(result.boxes)} objects detected")

                time.sleep(0.01)

            cap.release()
        except Exception as e:
            error = str(e)

        stopped = self.stop_video
        self.ui(lambda: self._video_done(page, frame_count, total_detections, stopped, error))

    def _video_done(self, page, frame_count, total_detections, stopped, error):
        self._ui_busy = False
        self._end_job(page)
        if page.mode == 'webcam' and page.save_btn:
            page.save_btn.configure(state=tk.DISABLED)

        if error:
            messagebox.showerror("Error", f"Video processing failed:\n{error}")
            self.update_status("Video processing failed", COLORS['error'])
        elif self._swapping:
            pass  # camera swap in progress; status handled there
        elif not stopped:
            self.update_status(
                f"✓ Video complete: {frame_count} frames, {total_detections} total detections",
                COLORS['success'])
            messagebox.showinfo("Complete",
                                f"Video processing complete!\n"
                                f"Frames: {frame_count}\n"
                                f"Total detections: {total_detections}")
        else:
            self.update_status("Video processing stopped", COLORS['warning'])

    def toggle_pause(self):
        """Play/pause toggle for video playback"""
        page = self.pages['video']
        if not self.is_processing:
            return
        self.paused = not self.paused
        if self.paused:
            page.pause_btn.configure(text="▶ Play")
            self.update_status("Video paused", COLORS['warning'])
        else:
            page.pause_btn.configure(text="⏸ Pause")
            self.update_status("Video playing", COLORS['accent'])

    def capture_frame(self):
        """Save the current annotated webcam frame instantly (no dialog)"""
        page = self.pages['webcam']
        if page.last_result is None:
            messagebox.showwarning("Warning", "No frame to capture yet!")
            return
        try:
            out_dir = Path('captures')
            out_dir.mkdir(exist_ok=True)
            file_path = out_dir / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            frame = page.last_result.copy()
            cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.update_status(f"📸 Captured: {file_path}", COLORS['success'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture frame:\n{str(e)}")

    def stop_detection(self):
        """Stop video processing"""
        self.paused = False
        self.stop_video = True
        self.update_status("Stopping...", COLORS['warning'])

    # ============================================================
    # Results panel
    # ============================================================
    def _clear_results(self, page):
        for widget in page.result_frame.winfo_children():
            widget.destroy()

    def show_no_results(self, page):
        self._clear_results(page)
        hint = {
            'image': "No detections yet\n\nSelect an image\nto run detection",
            'video': "No detections yet\n\nSelect a video\nto run detection",
            'webcam': "No detections yet\n\nStart the webcam\nto run detection",
            'folder': "No detections yet\n\nSelect a folder\nto run batch detection",
        }[page.mode]
        ctk.CTkLabel(page.result_frame, text=hint, font=F(12),
                     text_color=COLORS['fg_dim'], justify='center').pack(fill='both', pady=50)

    def _total_card(self, page, count, subtitle="Objects Detected"):
        card = ctk.CTkFrame(page.result_frame, fg_color=COLORS['bg_lighter'],
                            border_color=COLORS['accent'], border_width=2, corner_radius=10)
        card.pack(fill='x', pady=(0, 10), padx=4)
        ctk.CTkLabel(card, text=f"{count}", font=F(36, True),
                     text_color=COLORS['accent']).pack(pady=(12, 0))
        ctk.CTkLabel(card, text=subtitle, font=F(11),
                     text_color=COLORS['fg_dim']).pack(pady=(0, 12))

    def _section_header(self, page, text):
        ctk.CTkLabel(page.result_frame, text=text, font=F(12, True), anchor='w',
                     text_color=COLORS['accent']).pack(fill='x', padx=8, pady=(8, 2))

    def display_results(self, page, result, frame_num=None):
        """Display detection results in structured format"""
        self._clear_results(page)
        boxes = result.boxes

        if frame_num:
            ctk.CTkLabel(page.result_frame, text=f"FRAME {frame_num}", font=F(12, True),
                         text_color=COLORS['accent'], fg_color=COLORS['bg_lighter'],
                         corner_radius=8, height=36).pack(fill='x', pady=(0, 5), padx=4)

        self._total_card(page, len(boxes))

        if len(boxes) == 0:
            ctk.CTkLabel(page.result_frame, text="No objects detected", font=F(11),
                         text_color=COLORS['fg_dim']).pack(pady=20)
            return

        self._section_header(page, "📈 Summary by Class")

        class_counts = {}
        for box in boxes:
            class_name = result.names[int(box.cls[0])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            self.create_summary_card(page, class_name, count)

        ctk.CTkFrame(page.result_frame, fg_color=COLORS['border'], height=2).pack(
            fill='x', pady=12, padx=4)

        self._section_header(page, "🔍 Detailed Detections")
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            self.create_detection_card(page, i, class_name, conf, x1, y1, x2, y2)

    def display_folder_results(self, page, entries):
        """Analysis for the 2x2 grid: overall totals, class summary across all
        images, then one card per image (count, avg/max confidence, classes)."""
        self._clear_results(page)

        total = sum(len(r.boxes) for _, r in entries)
        all_confs = [float(b.conf[0]) for _, r in entries for b in r.boxes]
        avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

        self._total_card(page, total, f"Objects Detected in {len(entries)} Images")

        if total == 0:
            ctk.CTkLabel(page.result_frame, text="No objects detected", font=F(11),
                         text_color=COLORS['fg_dim']).pack(pady=20)
        else:
            # Overall stats row
            stats = ctk.CTkFrame(page.result_frame, fg_color=COLORS['bg_lighter'],
                                 corner_radius=8)
            stats.pack(fill='x', padx=4, pady=(0, 6))
            for label, value in (("Avg Confidence", f"{avg_conf * 100:.1f}%"),
                                 ("Max Confidence", f"{max(all_confs) * 100:.1f}%"),
                                 ("Avg / Image", f"{total / len(entries):.1f}")):
                col = ctk.CTkFrame(stats, fg_color='transparent')
                col.pack(side='left', expand=True, pady=8)
                ctk.CTkLabel(col, text=value, font=F(13, True),
                             text_color=COLORS['success']).pack()
                ctk.CTkLabel(col, text=label, font=F(9),
                             text_color=COLORS['fg_dim']).pack()

            self._section_header(page, "📈 Summary by Class")
            class_counts = {}
            for _, r in entries:
                for b in r.boxes:
                    n = r.names[int(b.cls[0])]
                    class_counts[n] = class_counts.get(n, 0) + 1
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                self.create_summary_card(page, class_name, count)

        ctk.CTkFrame(page.result_frame, fg_color=COLORS['border'], height=2).pack(
            fill='x', pady=12, padx=4)

        self._section_header(page, "🔍 Detailed Detections")
        for i, (name, r) in enumerate(entries, 1):
            self.create_image_card(page, i, name, r)

    def create_summary_card(self, page, class_name, count):
        """Create a summary card for each class"""
        card = ctk.CTkFrame(page.result_frame, fg_color=COLORS['bg_lighter'], corner_radius=8)
        card.pack(fill='x', pady=2, padx=4)
        ctk.CTkLabel(card, text=class_name.capitalize(), font=F(11, True),
                     anchor='w').pack(side='left', fill='x', expand=True, padx=12, pady=8)
        ctk.CTkLabel(card, text=str(count), font=F(12, True), width=40, corner_radius=6,
                     fg_color=COLORS['accent']).pack(side='right', padx=10, pady=8)

    def create_image_card(self, page, index, name, result):
        """Per-image analysis card used by folder mode"""
        boxes = result.boxes
        card = ctk.CTkFrame(page.result_frame, fg_color=COLORS['bg_lighter'],
                            border_color=COLORS['border'], border_width=1, corner_radius=8)
        card.pack(fill='x', pady=3, padx=4)

        head = ctk.CTkFrame(card, fg_color='transparent')
        head.pack(fill='x', padx=10, pady=(8, 2))
        ctk.CTkLabel(head, text=f"#{index}", font=F(10, True), width=32,
                     text_color=COLORS['accent']).pack(side='left')
        short = name if len(name) <= 26 else name[:23] + "..."
        ctk.CTkLabel(head, text=short, font=F(11, True), anchor='w').pack(
            side='left', fill='x', expand=True, padx=4)
        ctk.CTkLabel(head, text=str(len(boxes)), font=F(11, True), width=34, corner_radius=6,
                     fg_color=COLORS['accent']).pack(side='right')

        if not boxes:
            ctk.CTkLabel(card, text="No objects detected", font=F(10),
                         text_color=COLORS['fg_dim']).pack(anchor='w', padx=12, pady=(0, 8))
            return

        confs = [float(b.conf[0]) for b in boxes]
        avg = sum(confs) / len(confs)
        counts = {}
        for b in boxes:
            n = result.names[int(b.cls[0])]
            counts[n] = counts.get(n, 0) + 1
        cls_text = ", ".join(f"{n.capitalize()} ×{c}" for n, c in
                             sorted(counts.items(), key=lambda x: x[1], reverse=True))

        conf_color = (COLORS['success'] if avg > 0.7
                      else COLORS['warning'] if avg > 0.4 else COLORS['error'])
        row = ctk.CTkFrame(card, fg_color='transparent')
        row.pack(fill='x', padx=10)
        ctk.CTkLabel(row, text="Confidence:", font=F(10),
                     text_color=COLORS['fg_dim']).pack(side='left')
        bar = ctk.CTkProgressBar(row, width=100, height=8, fg_color=COLORS['bg'],
                                 progress_color=conf_color)
        bar.set(max(0.0, min(1.0, avg)))
        bar.pack(side='left', padx=6)
        ctk.CTkLabel(row, text=f"avg {avg * 100:.1f}% · max {max(confs) * 100:.1f}%",
                     font=F(10, True), text_color=conf_color).pack(side='left')

        ctk.CTkLabel(card, text=cls_text, font=F(10), text_color=COLORS['fg_dim'],
                     wraplength=320, justify='left', anchor='w').pack(
            anchor='w', padx=12, pady=(4, 8))

    def create_detection_card(self, page, index, class_name, conf, x1, y1, x2, y2):
        """Create a detection card for individual detection"""
        card = ctk.CTkFrame(page.result_frame, fg_color=COLORS['bg_lighter'],
                            border_color=COLORS['border'], border_width=1, corner_radius=8)
        card.pack(fill='x', pady=3, padx=4)

        header = ctk.CTkFrame(card, fg_color='transparent')
        header.pack(fill='x', padx=10, pady=(8, 2))
        ctk.CTkLabel(header, text=f"#{index}", font=F(10, True), width=32,
                     text_color=COLORS['accent']).pack(side='left')
        ctk.CTkLabel(header, text=class_name.capitalize(), font=F(11, True),
                     anchor='w').pack(side='left', fill='x', expand=True, padx=4)

        conf_pct = max(0.0, min(1.0, conf))
        conf_color = (COLORS['success'] if conf_pct > 0.7
                      else COLORS['warning'] if conf_pct > 0.4
                      else COLORS['error'])

        row = ctk.CTkFrame(card, fg_color='transparent')
        row.pack(fill='x', padx=10)
        ctk.CTkLabel(row, text="Confidence:", font=F(10),
                     text_color=COLORS['fg_dim']).pack(side='left')
        bar = ctk.CTkProgressBar(row, width=120, height=8, fg_color=COLORS['bg'],
                                 progress_color=conf_color)
        bar.set(conf_pct)
        bar.pack(side='left', padx=6)
        ctk.CTkLabel(row, text=f"{conf_pct * 100:.1f}%", font=F(10, True),
                     text_color=conf_color).pack(side='left')

        ctk.CTkLabel(card, text=f"Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                     font=F(10, family='Courier'), text_color=COLORS['fg_dim']).pack(
            anchor='w', padx=12, pady=(4, 8))

    # ============================================================
    # Save / Clear
    # ============================================================
    def save_result(self):
        """Save detection result (image, last video frame, or the 2x2 grid)"""
        page = self.active_page
        if page is None or page.last_result is None:
            messagebox.showwarning("Warning", "No result to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            cv2.imwrite(file_path, cv2.cvtColor(page.last_result, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", f"Result saved to:\n{file_path}")
            self.update_status(f"✓ Result saved: {Path(file_path).name}", COLORS['success'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result:\n{str(e)}")

    def clear_display(self):
        """Clear display and results of the active page"""
        page = self.active_page
        if self.is_processing:
            self.paused = False
            self.stop_video = True

        page.canvas.delete("all")
        page.placeholder_visible = True
        cw = page.canvas.winfo_width() or 800
        ch = page.canvas.winfo_height() or 600
        page.canvas_text = page.canvas.create_text(
            cw // 2, ch // 2, text=PLACEHOLDER, fill=COLORS['fg_dim'],
            font=('Arial', 16), justify=tk.CENTER)

        self.show_no_results(page)

        page.last_result = None
        if page.mode == 'folder':
            self.folder_path = None
            self.folder_images = []
            page.shuffle_btn.configure(state=tk.DISABLED)
        else:
            self.current_file = None
        page.file_label.configure(text="No file selected", text_color=COLORS['error'])
        if page.save_btn:
            page.save_btn.configure(state=tk.DISABLED)
        self.update_status("Display cleared", COLORS['fg_dim'])


def main():
    """Main application entry point"""
    app = YOLODetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()