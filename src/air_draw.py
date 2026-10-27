import math
import os

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "handwritten_digit_cnn.keras"
)

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

PROC_WIDTH = 320            # hand tracking runs on a smaller frame (faster)
PROC_HEIGHT = 240

DRAW_COLOR = 255
DRAW_THICKNESS = 8

MIN_DIGIT_WIDTH = 4
MIN_DIGIT_HEIGHT = 15

STROKE_RATIO = 0.11         # pen width as a fraction of digit size (MNIST is ~0.1)

SMOOTHING = 0.55            # 0-1: higher follows the finger faster, lower is smoother
MIN_MOVE = 2                # ignore finger jitter smaller than this (pixels)
MAX_JUMP = 260              # bigger jump = new stroke, not a long line
START_FRAMES = 2            # gesture must hold this long before ink appears
STOP_FRAMES = 3             # gesture must be lost this long before a stroke ends
EXTENDED_RATIO = 1.12       # finger tip vs. middle joint distance from wrist
FOLDED_RATIO = 1.05

PREDICTION_DISPLAY_TIME = 180


# ==========================================
# FINGER DETECTION
# ==========================================

def landmark_distance(a, b):

    return math.hypot(
        (a.x - b.x) * PROC_WIDTH,
        (a.y - b.y) * PROC_HEIGHT
    )


def is_drawing_gesture(hand):
    """Index finger extended, other three folded.

    Compares distance from the wrist instead of screen height, so it still
    works when the hand is tilted or rotated."""

    lm = hand.landmark
    wrist = lm[0]

    def ratio(tip, pip):

        return (
            landmark_distance(lm[tip], wrist)
            / max(landmark_distance(lm[pip], wrist), 1e-6)
        )

    return (
        ratio(8, 6) > EXTENDED_RATIO
        and ratio(12, 10) < FOLDED_RATIO
        and ratio(16, 14) < FOLDED_RATIO
        and ratio(20, 18) < FOLDED_RATIO
    )


# ==========================================
# CLEAN THE DRAWING
# ==========================================

def clean_pieces(ink):
    """Remove tiny specks and join pieces of the same digit
    (for example the top bar of a 5 that was drawn separately)."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink,
        connectivity=8
    )

    pieces = []

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        pieces.append(
            {"id": i, "x": x, "w": w, "size": max(w, h)}
        )

    if not pieces:
        return ink

    biggest = max(p["size"] for p in pieces)

    pieces = [
        p for p in pieces
        if p["size"] >= 0.2 * biggest
    ]

    pieces.sort(key=lambda p: p["x"])

    cleaned = np.zeros_like(ink)

    for p in pieces:
        cleaned[labels == p["id"]] = 255

    # Pieces stacked above/below each other belong to one digit
    for prev, cur in zip(pieces, pieces[1:]):

        overlap = (
            min(prev["x"] + prev["w"], cur["x"] + cur["w"])
            - cur["x"]
        )

        if overlap > 0.5 * min(prev["w"], cur["w"]):

            pa = np.column_stack(np.where(labels == prev["id"]))
            pb = np.column_stack(np.where(labels == cur["id"]))

            pa = pa[::max(1, len(pa) // 300)]
            pb = pb[::max(1, len(pb) // 300)]

            distance = np.linalg.norm(
                pa[:, None, :].astype(np.float32)
                - pb[None, :, :].astype(np.float32),
                axis=2
            )

            i, j = np.unravel_index(
                np.argmin(distance),
                distance.shape
            )

            cv2.line(
                cleaned,
                (int(pa[i][1]), int(pa[i][0])),
                (int(pb[j][1]), int(pb[j][0])),
                255,
                DRAW_THICKNESS
            )

    return cleaned


# ==========================================
# FIND SEPARATE DIGITS
# ==========================================

def find_digit_masks(canvas):
    """Return one clean mask per digit, ordered left to right."""

    ink = (canvas > 0).astype(np.uint8) * 255

    if not np.any(ink):
        return []

    # Bridge tiny gaps left by tracking dropouts
    gap = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, gap)

    ink = clean_pieces(ink)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink,
        connectivity=8
    )

    regions = []

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if w < MIN_DIGIT_WIDTH or h < MIN_DIGIT_HEIGHT or area < 30:
            continue

        # Only this digit's own pixels, never a neighbour's
        mask = (
            (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        )

        regions.append((x, mask))

    regions.sort(key=lambda region: region[0])

    return [mask for _, mask in regions]


# ==========================================
# DIGIT PREPROCESSING (MNIST style)
# ==========================================

def preprocess_digit(mask):
    """Returns (cnn_input, preview_28x28) or (None, None)."""

    if cv2.findNonZero(mask) is None:
        return None, None

    h, w = mask.shape

    # The pen is thin compared to the digit. Shrinking to 20 px would erase
    # the stroke, so thicken it to MNIST proportions first.
    thickness = max(
        DRAW_THICKNESS,
        int(round(STROKE_RATIO * max(h, w)))
    )

    k = thickness - DRAW_THICKNESS + 1

    pad = k + 4

    mask = cv2.copyMakeBorder(
        mask, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    if k > 1:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )

        mask = cv2.dilate(mask, kernel)

    # Tight bounding box
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))

    mask = mask[y:y + h, x:x + w]

    # Fit inside 20x20, keep aspect ratio
    scale = 20.0 / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    mask = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    image = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    image[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = mask

    # MNIST digits are centred by centre of mass, not by bounding box
    moments = cv2.moments(image)

    if moments["m00"] > 0:

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        shift = np.float32(
            [[1, 0, 14 - cx],
             [0, 1, 14 - cy]]
        )

        image = cv2.warpAffine(
            image,
            shift,
            (28, 28)
        )

    preview = image.copy()

    cnn_input = (
        image.astype("float32") / 255.0
    ).reshape(1, 28, 28, 1)

    return cnn_input, preview


# ==========================================
# PREDICT MULTIPLE DIGITS
# ==========================================

def predict_number(model, canvas):

    masks = find_digit_masks(canvas)

    if not masks:

        print("No digits detected.")

        return None

    predictions = []
    previews = []

    for mask in masks:

        processed, preview = preprocess_digit(mask)

        if processed is None:
            continue

        probabilities = model.predict(
            processed,
            verbose=0
        )[0]

        top_indices = np.argsort(probabilities)[::-1][:3]

        prediction = int(top_indices[0])

        predictions.append(
            {
                "digit": prediction,
                "confidence": float(probabilities[prediction]) * 100,
                "top_three": [
                    (int(i), float(probabilities[i]) * 100)
                    for i in top_indices
                ]
            }
        )

        previews.append(preview)

    if not predictions:

        print("Unable to recognize digits.")

        return None

    number = "".join(str(item["digit"]) for item in predictions)

    print("\n===================================")
    print("       AIR DRAWING RESULT")
    print("===================================")
    print("Predicted Number:", number)

    for index, item in enumerate(predictions, start=1):

        print(f"\nDigit {index}: {item['digit']}")
        print(f"Confidence: {item['confidence']:.2f}%")
        print("Top 3 predictions:")

        for digit_number, confidence in item["top_three"]:
            print(f"  {digit_number}: {confidence:.2f}%")

    print("===================================\n")

    return number, predictions, previews


# ==========================================
# MAIN
# ==========================================

def main():

    if not os.path.exists(MODEL_PATH):

        print("Model not found:", MODEL_PATH)

        raise SystemExit

    model = tf.keras.models.load_model(MODEL_PATH)

    print("CNN model loaded successfully.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Unable to access webcam.")

        hands.close()

        raise SystemExit

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)

    previous_point = None
    smooth_point = None
    gesture_frames = 0
    idle_frames = 0

    predicted_number = ""
    previews = []
    prediction_display_counter = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print("Unable to access webcam.")

            break

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))

        small = cv2.resize(
            frame,
            (PROC_WIDTH, PROC_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        results = hands.process(
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        )

        gesture = False
        raw_point = None

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = is_drawing_gesture(hand)

            tip = hand.landmark[8]

            raw_point = (
                tip.x * CANVAS_WIDTH,
                tip.y * CANVAS_HEIGHT
            )

        # Debounce the gesture: no accidental dots, no broken strokes
        if gesture:

            gesture_frames += 1
            idle_frames = 0

        else:

            idle_frames += 1
            gesture_frames = 0

            if idle_frames >= STOP_FRAMES:

                previous_point = None
                smooth_point = None

        drawing_active = gesture and gesture_frames >= START_FRAMES

        if drawing_active:

            # A big jump is a tracking glitch or a new stroke
            if smooth_point is not None and math.hypot(
                raw_point[0] - smooth_point[0],
                raw_point[1] - smooth_point[1]
            ) > MAX_JUMP:

                smooth_point = None
                previous_point = None

            if smooth_point is None:

                smooth_point = raw_point

            else:

                smooth_point = (
                    SMOOTHING * raw_point[0]
                    + (1 - SMOOTHING) * smooth_point[0],
                    SMOOTHING * raw_point[1]
                    + (1 - SMOOTHING) * smooth_point[1]
                )

            current_point = (
                int(round(smooth_point[0])),
                int(round(smooth_point[1]))
            )

            if previous_point is None:

                previous_point = current_point

            elif math.hypot(
                current_point[0] - previous_point[0],
                current_point[1] - previous_point[1]
            ) >= MIN_MOVE:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    DRAW_COLOR,
                    DRAW_THICKNESS,
                    cv2.LINE_AA
                )

                previous_point = current_point

        if raw_point is not None:

            cv2.circle(
                frame,
                (int(raw_point[0]), int(raw_point[1])),
                9,
                (0, 255, 0) if drawing_active else (255, 120, 80),
                2,
                cv2.LINE_AA
            )

        # ---------- webcam panel ----------

        cv2.putText(
            frame, "WEBCAM", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "Index finger only = draw", (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        # ---------- drawing panel ----------

        drawing_panel = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            drawing_panel, "DRAWING CANVAS", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        if drawing_active:
            status = "DRAWING"
        elif raw_point is not None:
            status = "READY"
        else:
            status = "NO HAND"

        cv2.putText(
            drawing_panel, status, (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        if prediction_display_counter > 0:

            cv2.putText(
                drawing_panel,
                "Prediction: " + predicted_number,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            # What the CNN actually received, one 28x28 tile per digit
            x_pos = 20

            for preview in previews[:9]:

                tile = cv2.cvtColor(
                    cv2.resize(preview, (56, 56), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR
                )

                drawing_panel[100:156, x_pos:x_pos + 56] = tile

                x_pos += 62

            prediction_display_counter -= 1

        display = np.hstack((frame, drawing_panel))

        cv2.putText(
            display,
            "P: Predict Number | C: Clear | Q: Quit",
            (CANVAS_WIDTH + 20, 440),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        cv2.imshow("Air Drawing - Digit Recognition", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):

            if not np.any(canvas):

                print("Draw a number first.")

            else:

                result = predict_number(model, canvas)

                if result is not None:

                    predicted_number, _, previews = result

                    prediction_display_counter = PREDICTION_DISPLAY_TIME

                previous_point = None
                smooth_point = None

        elif key == ord("c"):

            canvas[:] = 0

            previous_point = None
            smooth_point = None
            predicted_number = ""
            previews = []
            prediction_display_counter = 0

            print("Canvas cleared.")

        elif key == ord("q"):

            break

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print("Air Drawing application closed.")


if __name__ == "__main__":

    main()