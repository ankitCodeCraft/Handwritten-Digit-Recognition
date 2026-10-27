import math
import os

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

from PIL import Image
import mediapipe as mp


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "models",
    "handwritten_digit_cnn.keras"
)


# ==================================================
# LOAD CNN MODEL
# ==================================================

model = tf.keras.models.load_model(MODEL_PATH)


# ==================================================
# EXTRACT IMAGE
# ==================================================

def extract_image(image):

    if image is None:
        return None

    if isinstance(image, dict):
        image = image.get("composite")

    if isinstance(image, np.ndarray):

        if image.size == 0:
            return None

        image = Image.fromarray(
            image.astype("uint8")
        )

    if not isinstance(image, Image.Image):
        return None

    return image.convert("RGB")


# ==================================================
# CREATE BINARY IMAGE
# ==================================================

def create_binary_image(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Convert handwriting into white
    # digits on black background
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


# ==================================================
# PREPROCESS ONE DIGIT
# ==================================================

def prepare_digit(digit_image):

    digit_image = np.asarray(
        digit_image,
        dtype=np.uint8
    )

    if digit_image.size == 0:
        return None

    # Find digit boundaries
    coords = np.argwhere(
        digit_image > 0
    )

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit_image = digit_image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    height, width = digit_image.shape

    if height == 0 or width == 0:
        return None

    # Resize to fit inside 20x20
    scale = 20.0 / max(
        height,
        width
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    digit_image = cv2.resize(
        digit_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # MNIST canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (
        28 - new_width
    ) // 2

    y_offset = (
        28 - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit_image

    # Normalize
    canvas = canvas.astype(
        np.float32
    ) / 255.0

    return canvas.reshape(
        1, 28, 28, 1
    )


# ==================================================
# SEGMENT DIGITS
# ==================================================

def segment_digits(image):

    binary = create_binary_image(
        image
    )

    height, width = binary.shape

    # Remove small noise
    kernel = np.ones(
        (2, 2),
        dtype=np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Connected components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for i in range(1, count):

        x = stats[
            i, cv2.CC_STAT_LEFT
        ]

        y = stats[
            i, cv2.CC_STAT_TOP
        ]

        w = stats[
            i, cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i, cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            i, cv2.CC_STAT_AREA
        ]

        # Ignore tiny components
        if area < 20:
            continue

        if w < 3 or h < 5:
            continue

        # Ignore large background components
        if w > width * 0.95:
            continue

        if h > height * 0.95:
            continue

        components.append(
            (x, y, w, h)
        )

    if not components:
        return []

    # Left-to-right ordering
    components.sort(
        key=lambda item: item[0]
    )

    digit_regions = []

    for x, y, w, h in components:

        digit_region = binary[
            y:y + h,
            x:x + w
        ]

        if digit_region.size == 0:
            continue

        digit_regions.append(
            (x, digit_region)
        )

    return [
        region
        for _, region in digit_regions
    ]


# ==================================================
# RECOGNIZE NUMBER
# ==================================================

def recognize_number(image):

    image = extract_image(
        image
    )

    if image is None:

        return (
            {},
            "Please upload an image or draw digits."
        )

    digit_regions = segment_digits(
        image
    )

    if not digit_regions:

        return (
            {},
            "No digits detected. Please try again."
        )

    recognized_digits = []
    confidence_values = []

    first_probabilities = None

    for digit_region in digit_regions:

        processed = prepare_digit(
            digit_region
        )

        if processed is None:
            continue

        probabilities = model.predict(
            processed,
            verbose=0
        )[0]

        digit = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[digit]
        ) * 100

        recognized_digits.append(
            str(digit)
        )

        confidence_values.append(
            confidence
        )

        if first_probabilities is None:

            first_probabilities = probabilities

    if not recognized_digits:

        return (
            {},
            "Unable to recognize digits."
        )

    # Complete number
    number = "".join(
        recognized_digits
    )

    # Individual confidence
    confidence_text = "\n".join(
        [
            f"Digit {i + 1}: {digit} "
            f"({confidence:.2f}%)"
            for i, (digit, confidence)
            in enumerate(
                zip(
                    recognized_digits,
                    confidence_values
                )
            )
        ]
    )

    message = (
        f"Recognized Number: {number}\n\n"
        f"{confidence_text}"
    )

    # Probability display for first digit
    results = {
        str(i): float(value)
        for i, value in enumerate(
            first_probabilities
        )
    }

    return results, message


# ==================================================
# CLEAR CANVAS
# ==================================================

def clear_canvas():

    blank = Image.new(
        "RGB",
        (560, 280),
        "white"
    )

    return {
        "background": blank,
        "layers": [],
        "composite": blank
    }


# ==================================================
# AIR DRAWING (Gradio webcam stream)
# ==================================================

AIR_WIDTH = 640            # drawing canvas (kept at full size for quality)
AIR_HEIGHT = 480
PROC_WIDTH = 320           # hand tracking runs on a smaller frame (less lag)
PROC_HEIGHT = 240
AIR_DRAW_THICKNESS = 10

SMOOTHING = 0.55           # 0-1, higher follows the finger faster, lower is smoother
MIN_MOVE = 2               # ignore finger jitter smaller than this (canvas px)
MAX_JUMP = 260             # a bigger jump starts a new stroke instead of a long line
START_FRAMES = 2           # gesture must hold this many frames before ink appears
STOP_FRAMES = 3            # gesture must be lost this many frames before a stroke ends
EXTENDED_RATIO = 1.12      # fingertip vs. middle joint distance from wrist (extended)
FOLDED_RATIO = 1.05        # same ratio for folded fingers

_air_mp_hands = mp.solutions.hands
_air_draw_utils = mp.solutions.drawing_utils

_AIR_TRACKERS = {}         # one hand tracker per browser session
_AIR_PAUSED = set()        # sessions whose stream is paused during CNN inference


def _air_new_state():
    return {
        "canvas": np.zeros((AIR_HEIGHT, AIR_WIDTH), dtype=np.uint8),
        "previous": None,      # last point actually drawn
        "smooth": None,        # smoothed fingertip position
        "gesture_frames": 0,
        "idle_frames": 0,
    }


def _air_reset_stroke(state):
    state["previous"] = None
    state["smooth"] = None
    state["gesture_frames"] = 0
    state["idle_frames"] = 0


def _air_sid(request):
    return getattr(request, "session_hash", None) or "default"


def _air_get_tracker(sid):
    tracker = _AIR_TRACKERS.get(sid)
    if tracker is None:
        tracker = _air_mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,            # fastest hand model
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        _AIR_TRACKERS[sid] = tracker
    return tracker


def _air_dist(a, b):
    return math.hypot(
        (a.x - b.x) * PROC_WIDTH,
        (a.y - b.y) * PROC_HEIGHT
    )


def _air_is_drawing_gesture(hand):
    """Index finger extended, other three fingers folded.
    Uses distance from the wrist, so it works at any hand angle."""
    lm = hand.landmark
    wrist = lm[0]

    def ratio(tip, pip):
        return _air_dist(lm[tip], wrist) / max(_air_dist(lm[pip], wrist), 1e-6)

    return (
        ratio(8, 6) > EXTENDED_RATIO
        and ratio(12, 10) < FOLDED_RATIO
        and ratio(16, 14) < FOLDED_RATIO
        and ratio(20, 18) < FOLDED_RATIO
    )


def air_process_frame(frame, state, request: gr.Request):
    """Track the finger and draw onto the persistent canvas."""
    sid = _air_sid(request)

    if state is None:
        state = _air_new_state()

    if frame is None:
        return gr.update(), state, "Camera frame not received."

    # Prediction running: skip tracking so the CNN gets the CPU
    if sid in _AIR_PAUSED:
        return gr.update(), state, gr.update()

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return gr.update(), state, "Invalid webcam frame."

    frame = cv2.resize(
        frame[:, :, :3], (PROC_WIDTH, PROC_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    frame = cv2.flip(frame, 1)

    canvas = state["canvas"]
    result = _air_get_tracker(sid).process(frame)

    gesture = False
    raw_point = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        _air_draw_utils.draw_landmarks(
            frame, hand, _air_mp_hands.HAND_CONNECTIONS
        )
        gesture = _air_is_drawing_gesture(hand)
        tip = hand.landmark[8]
        raw_point = (tip.x * AIR_WIDTH, tip.y * AIR_HEIGHT)

    # Gesture debouncing: no accidental dots, no broken strokes on flicker
    if gesture:
        state["gesture_frames"] += 1
        state["idle_frames"] = 0
    else:
        state["idle_frames"] += 1
        state["gesture_frames"] = 0
        if state["idle_frames"] >= STOP_FRAMES:
            state["previous"] = None
            state["smooth"] = None

    drawing = gesture and state["gesture_frames"] >= START_FRAMES
    cursor = None

    if drawing:
        smooth = state["smooth"]

        # Big jump = tracking glitch or a new stroke, so don't connect it
        if smooth is not None and math.hypot(
            raw_point[0] - smooth[0], raw_point[1] - smooth[1]
        ) > MAX_JUMP:
            smooth = None
            state["previous"] = None

        if smooth is None:
            smooth = raw_point
        else:
            smooth = (
                SMOOTHING * raw_point[0] + (1 - SMOOTHING) * smooth[0],
                SMOOTHING * raw_point[1] + (1 - SMOOTHING) * smooth[1],
            )
        state["smooth"] = smooth

        point = (int(round(smooth[0])), int(round(smooth[1])))
        cursor = point
        previous = state["previous"]

        if previous is None:
            state["previous"] = point        # stroke starts; no stray dot
        elif math.hypot(point[0] - previous[0], point[1] - previous[1]) >= MIN_MOVE:
            cv2.line(
                canvas, previous, point, 255,
                AIR_DRAW_THICKNESS, cv2.LINE_AA
            )
            state["previous"] = point
    elif raw_point is not None:
        cursor = (int(raw_point[0]), int(raw_point[1]))

    # ----- Display: camera on the left, persistent drawing on the right -----
    scale_x = PROC_WIDTH / AIR_WIDTH
    scale_y = PROC_HEIGHT / AIR_HEIGHT
    small = cv2.resize(
        canvas, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA
    )
    board = np.full((PROC_HEIGHT, PROC_WIDTH, 3), 250, dtype=np.uint8)
    board[small > 60] = (45, 40, 140)

    if cursor is not None:
        color = (40, 200, 90) if drawing else (79, 70, 229)
        cx = int(cursor[0] * scale_x)
        cy = int(cursor[1] * scale_y)
        cv2.circle(frame, (cx, cy), 7, color, 2, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), 6, color, 2, cv2.LINE_AA)

    if drawing:
        status = "Drawing: index finger up"
    elif raw_point is not None:
        status = "Ready: raise only your index finger to draw"
    else:
        status = "No hand detected"

    return np.hstack((frame, board)), state, status


def _air_clean_pieces(ink):
    """Drop stray specks and join pieces that belong to the same digit
    (for example the top bar of a 5 that was drawn separately)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    comps = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append({"id": i, "x": x, "w": w, "size": max(w, h)})

    if not comps:
        return ink

    biggest = max(c["size"] for c in comps)
    comps = [c for c in comps if c["size"] >= 0.2 * biggest]
    comps.sort(key=lambda c: c["x"])

    out = np.zeros_like(ink)
    for c in comps:
        out[labels == c["id"]] = 255

    for prev, cur in zip(comps, comps[1:]):
        overlap = min(prev["x"] + prev["w"], cur["x"] + cur["w"]) - cur["x"]
        if overlap > 0.5 * min(prev["w"], cur["w"]):
            pa = np.column_stack(np.where(labels == prev["id"]))
            pb = np.column_stack(np.where(labels == cur["id"]))
            pa = pa[::max(1, len(pa) // 300)]
            pb = pb[::max(1, len(pb) // 300)]
            dist = np.linalg.norm(
                pa[:, None, :].astype(np.float32) - pb[None, :, :].astype(np.float32),
                axis=2
            )
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            cv2.line(
                out,
                (int(pa[i][1]), int(pa[i][0])),
                (int(pb[j][1]), int(pb[j][0])),
                255, AIR_DRAW_THICKNESS
            )

    return out


def air_prepare_image(canvas):
    """Turn the air canvas into the white-background / dark-ink image that
    recognize_number() expects, with strokes thick enough for the CNN."""
    ink = (canvas > 0).astype(np.uint8) * 255

    if not np.any(ink):
        return None

    # Bridge tiny gaps left by tracking dropouts so one digit stays one piece
    gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, gap)
    ink = _air_clean_pieces(ink)

    # Crop to the drawing
    ys, xs = np.where(ink > 0)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape

    # Thin strokes vanish when a big digit is shrunk to 20x20, so thicken
    # the pen to roughly a tenth of the digit height (like MNIST)
    target = max(AIR_DRAW_THICKNESS, int(h * 0.10))
    k = target - AIR_DRAW_THICKNESS + 1

    pad = max(20, int(0.2 * max(h, w))) + k
    ink = cv2.copyMakeBorder(
        ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )

    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ink = cv2.dilate(ink, kernel)

    return Image.fromarray(255 - ink).convert("RGB")


def air_predict(state, request: gr.Request):
    sid = _air_sid(request)
    _AIR_PAUSED.add(sid)          # pause stream processing while the CNN runs

    try:
        if state is None or not np.any(state["canvas"]):
            return "Draw a digit first.", {}, None

        image = air_prepare_image(state["canvas"].copy())
        probabilities, message = recognize_number(image)
        _air_reset_stroke(state)
        return message, probabilities, image

    finally:
        _AIR_PAUSED.discard(sid)


def air_clear(state, request: gr.Request):
    if state is None:
        state = _air_new_state()

    state["canvas"][:] = 0        # in place, so an in-flight frame can't bring it back
    _air_reset_stroke(state)

    return (
        state,
        "Canvas cleared. Raise your index finger to draw.",
        "",
        {},
        None
    )


def air_cleanup(request: gr.Request = None):
    if request is None:
        return
    sid = _air_sid(request)
    tracker = _AIR_TRACKERS.pop(sid, None)
    if tracker is not None:
        tracker.close()
    _AIR_PAUSED.discard(sid)


AIR_TIPS_HTML = """
<div class="tips">
  <div class="tip"><b>Index finger only</b>Fold the other three fingers to draw.</div>
  <div class="tip"><b>Open hand to pause</b>Lift your fingers to move to the next digit.</div>
  <div class="tip"><b>Write big and slow</b>Large, steady digits are read more accurately.</div>
</div>
"""


# ==================================================
# UI THEME + STYLES  (visual layer only)
# ==================================================

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[
        gr.themes.GoogleFont("Plus Jakarta Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
    --ink: #121a33;
    --muted: #5d6784;
    --line: #dfe4f1;
    --paper: #f3f6fc;
    --brand: #4f46e5;
    --brand-2: #0ea5a0;
    --mark: #ffe27a;
}

body, .gradio-container {
    background-color: var(--paper) !important;
    background-image:
        linear-gradient(rgba(79,70,229,.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,70,229,.055) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--ink);
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 26px 28px 36px !important;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 40px 44px;
    margin-bottom: 22px;
    color: #fff;
    background:
        radial-gradient(circle at 88% 12%, rgba(14,165,160,.35), transparent 42%),
        linear-gradient(130deg, #111833 0%, #232a66 60%, #4338ca 100%);
    box-shadow: 0 20px 50px rgba(30,36,100,.22);
}
.hero-grid { display: flex; gap: 32px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
.hero h1 {
    margin: 0 0 12px !important;
    color: #fff !important;
    font-size: clamp(30px, 4vw, 48px) !important;
    font-weight: 800 !important;
    letter-spacing: -1.4px;
    line-height: 1.08 !important;
}
.hero p { margin: 0 !important; max-width: 480px; color: #cfd6f6 !important; font-size: 15.5px; line-height: 1.6; }
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; }
.hero-chips span {
    padding: 6px 13px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
    background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.22); color: #eef0ff;
}
.hero-digits {
    font-family: 'Caveat', cursive;
    font-size: clamp(54px, 7vw, 92px);
    font-weight: 700;
    line-height: .95;
    letter-spacing: 6px;
    text-align: right;
    color: #fff;
    transform: rotate(-4deg);
    text-shadow: 0 4px 24px rgba(0,0,0,.25);
}
.hero-digits em {
    font-style: normal;
    background: linear-gradient(transparent 58%, rgba(255,226,122,.55) 58%);
    padding: 0 4px;
}
.hero-digits small { display: block; font-size: 15px; letter-spacing: 1px; color: #b9c2f0; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 500; margin-top: 8px; }

/* ---------- Panels ---------- */
.workspace { gap: 20px !important; align-items: stretch !important; }
.panel {
    background: rgba(255,255,255,.96) !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(24,34,80,.06) !important;
}
.panel-title h3 { margin: 0 0 4px !important; font-size: 19px !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -.3px; }
.panel-title p { margin: 0 0 6px !important; color: var(--muted) !important; font-size: 13.5px !important; line-height: 1.55; }

/* ---------- Tabs ---------- */
.tab-nav, div[role="tablist"] { border-bottom: 1px solid var(--line) !important; margin-bottom: 14px; }
.tab-nav button, button[role="tab"] { font-weight: 600 !important; font-size: 14.5px !important; padding: 10px 18px !important; color: var(--muted) !important; }
.tab-nav button.selected, button[role="tab"][aria-selected="true"] { color: var(--brand) !important; }

/* ---------- Inputs ---------- */
.gradio-container .block, .gradio-container .form { border-radius: 16px !important; }
.canvas-wrap { border-radius: 18px !important; overflow: hidden; border: 2px dashed #c8cff0 !important; transition: border-color .2s ease; }
.canvas-wrap:hover, .canvas-wrap:focus-within { border-color: var(--brand) !important; }

/* ---------- Result ---------- */
.result-box textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 1.6 !important;
    color: var(--ink) !important;
    background: linear-gradient(180deg, #fffdf2, #fff8d6) !important;
    border: 1px solid #f1e3a0 !important;
    border-radius: 16px !important;
}

/* ---------- Buttons ---------- */
button.primary {
    min-height: 50px !important;
    border: 0 !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: .1px;
    color: #fff !important;
    background: linear-gradient(135deg, #4f46e5, #6d5df6) !important;
    box-shadow: 0 10px 22px rgba(79,70,229,.32) !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
button.primary:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(79,70,229,.4) !important; }
button.primary:active { transform: translateY(0); }
button.secondary {
    min-height: 50px !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    background: #fff !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}
button:focus-visible { outline: 3px solid rgba(79,70,229,.45) !important; outline-offset: 2px; }

/* ---------- Tips ---------- */
.tips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
.tip { background: #f6f8fe; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; font-size: 12.5px; line-height: 1.5; color: var(--muted); }
.tip b { display: block; color: var(--ink); font-size: 13px; margin-bottom: 2px; }

/* ---------- Air drawing ---------- */
.air-card { background: linear-gradient(135deg, #eef0ff, #e6f8f6); border: 1px solid var(--line); border-radius: 18px; padding: 6px 22px; }

/* ---------- Footer ---------- */
.site-footer { text-align: center; color: #7b8499; font-size: 12.5px; padding: 26px 0 0; }
.site-footer b { color: var(--brand); font-weight: 600; }

@media (max-width: 900px) {
    .gradio-container { padding: 14px !important; }
    .hero { padding: 26px 22px; border-radius: 20px; }
    .hero-digits { text-align: left; }
    .tips { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
"""

TIPS_HTML = """
<div class="tips">
  <div class="tip"><b>Leave gaps</b>Keep a small space between digits so each one is read separately.</div>
  <div class="tip"><b>One line</b>Write from left to right in a single horizontal line.</div>
  <div class="tip"><b>Bold strokes</b>Dark, thick strokes on a light background work best.</div>
</div>
"""


FORCE_LIGHT_JS = """
() => {
    const apply = () => document.body.classList.remove('dark');
    apply();
    new MutationObserver(apply).observe(
        document.body, { attributes: true, attributeFilter: ['class'] }
    );
}
"""


# ==================================================
# MODERN GRADIO INTERFACE
# ==================================================

with gr.Blocks(
    title="Handwritten Digit Recognition | CNN",
    theme=THEME,
    css=CUSTOM_CSS,
    js=FORCE_LIGHT_JS
) as demo:

    gr.HTML("""
    <section class="hero">
      <div class="hero-grid">
        <div>
          <h1>Handwritten Digit<br>Recognition</h1>
          <p>Write or upload a number and your trained CNN reads it digit by digit, with a confidence score for each one.</p>
          <div class="hero-chips">
            <span>Upload or draw</span>
            <span>Multi-digit numbers</span>
            <span>Live confidence scores</span>
          </div>
        </div>
        <div class="hero-digits">
          <em>4 2 7</em> 1 9
          <small>your handwriting, read by a CNN</small>
        </div>
      </div>
    </section>
    """)

    with gr.Tabs():

        with gr.Tab("Upload image"):

            with gr.Row(elem_classes=["workspace"]):

                with gr.Column(scale=6, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Add your image
                    Upload a clear photo or scan of a handwritten number. Dark digits on a light background work best.
                    """, elem_classes=["panel-title"])

                    image_input = gr.Image(
                        type="pil",
                        label="Image canvas",
                        height=390,
                        sources=["upload", "clipboard"],
                        show_download_button=False,
                        elem_classes=["canvas-wrap"]
                    )

                    upload_button = gr.Button(
                        "Recognize number",
                        variant="primary",
                        elem_classes=["primary"]
                    )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Result
                    The predicted number and the confidence for each digit appear here.
                    """, elem_classes=["panel-title"])

                    upload_result = gr.Textbox(
                        label="Predicted number",
                        lines=7,
                        interactive=False,
                        elem_classes=["result-box"],
                        placeholder="Your result will appear here after recognition."
                    )

                    upload_prediction = gr.Label(
                        num_top_classes=5,
                        label="Probability distribution · first digit"
                    )

                    gr.HTML(TIPS_HTML)

            upload_button.click(
                fn=recognize_number,
                inputs=image_input,
                outputs=[upload_prediction, upload_result]
            )

        with gr.Tab("Draw digits"):

            with gr.Row(elem_classes=["workspace"]):

                with gr.Column(scale=6, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Draw your number
                    Use the brush and write from left to right. Example: **12345**
                    """, elem_classes=["panel-title"])

                    canvas_input = gr.ImageEditor(
                        type="pil",
                        label="Drawing board",
                        image_mode="RGB",
                        canvas_size=(700, 340),
                        layers=False,
                        transforms=[],
                        brush=gr.Brush(
                            colors=["#111827"],
                            default_size=18
                        ),
                        elem_classes=["canvas-wrap"]
                    )

                    with gr.Row():
                        canvas_predict_button = gr.Button(
                            "Recognize number",
                            variant="primary",
                            elem_classes=["primary"]
                        )
                        canvas_clear_button = gr.Button(
                            "Clear board",
                            variant="secondary",
                            elem_classes=["secondary"]
                        )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Result
                    The model reads each digit and reports its confidence.
                    """, elem_classes=["panel-title"])

                    canvas_result = gr.Textbox(
                        label="Predicted number",
                        lines=7,
                        interactive=False,
                        elem_classes=["result-box"],
                        placeholder="Draw digits, then select Recognize number."
                    )

                    canvas_prediction = gr.Label(
                        num_top_classes=5,
                        label="Probability distribution · first digit"
                    )

                    gr.HTML(TIPS_HTML)

            canvas_predict_button.click(
                fn=recognize_number,
                inputs=canvas_input,
                outputs=[canvas_prediction, canvas_result]
            )

            canvas_clear_button.click(
                fn=clear_canvas,
                inputs=None,
                outputs=canvas_input
            )

        with gr.Tab("Air drawing"):

            air_state = gr.State(_air_new_state())

            with gr.Row(elem_classes=["workspace"]):

                with gr.Column(scale=6, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Draw in the air
                    Allow camera access, then raise only your index finger to write. Open your hand to pause between digits.
                    """, elem_classes=["panel-title"])

                    air_camera = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        mirror_webcam=False,
                        label="Webcam",
                        height=240,
                        elem_classes=["canvas-wrap"]
                    )

                    air_stream_view = gr.Image(
                        type="numpy",
                        format="jpeg",
                        label="Hand tracking and air canvas",
                        height=300,
                        interactive=False,
                        show_download_button=False
                    )

                    air_status = gr.Textbox(
                        label="Gesture status",
                        value="Start the webcam and raise your index finger.",
                        interactive=False
                    )

                    with gr.Row():
                        air_predict_button = gr.Button(
                            "Recognize number",
                            variant="primary",
                            elem_classes=["primary"]
                        )
                        air_clear_button = gr.Button(
                            "Clear canvas",
                            variant="secondary",
                            elem_classes=["secondary"]
                        )

                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("""
                    ### Result
                    The model reads each air-drawn digit and reports its confidence.
                    """, elem_classes=["panel-title"])

                    air_result = gr.Textbox(
                        label="Predicted number",
                        lines=7,
                        interactive=False,
                        elem_classes=["result-box"],
                        placeholder="Draw in the air, then select Recognize number."
                    )

                    air_probabilities = gr.Label(
                        num_top_classes=5,
                        label="Probability distribution · first digit"
                    )

                    air_preview = gr.Image(
                        type="pil",
                        label="What the model sees",
                        height=150,
                        interactive=False,
                        show_download_button=False
                    )

                    gr.HTML(AIR_TIPS_HTML)

            air_camera.stream(
                fn=air_process_frame,
                inputs=[air_camera, air_state],
                outputs=[air_stream_view, air_state, air_status],
                stream_every=0.1,
                time_limit=600,
                concurrency_limit=1,
                trigger_mode="always_last",
                show_progress="hidden"
            )

            air_predict_button.click(
                fn=air_predict,
                inputs=air_state,
                outputs=[air_result, air_probabilities, air_preview]
            )

            air_clear_button.click(
                fn=air_clear,
                inputs=air_state,
                outputs=[air_state, air_status, air_result,
                         air_probabilities, air_preview]
            )

    gr.HTML("""
    <div class="site-footer">
      Built with <b>TensorFlow</b> · <b>OpenCV</b> · <b>Gradio</b> &nbsp;|&nbsp; Handwritten Digit Recognition
    </div>
    """)


demo.unload(air_cleanup)


# ==================================================
# RUN APP
# ==================================================

if __name__ == "__main__":

    demo.launch()