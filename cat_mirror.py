import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Settings (tweak if needed) ---
MOUTH_OPEN_THRESH = 0.02   # distance threshold between upper/lower lip for "open"
ELBOW_BUFFER = 0.02        # small buffer so wrist must be reasonably above elbow
MOUTH_SHUT_TIMEOUT = 1.0   # seconds before reverting to normal after mouth shut

# --- Step 1: Initialize ---
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
pose_detector = mp_pose.Pose()

# Path to cats folder
CAT_DIR = os.path.join(os.path.dirname(__file__), "cats")

# Load cat images
cat_images = {}
if os.path.exists(CAT_DIR):
    for f in os.listdir(CAT_DIR):
        if f.lower().endswith(".jpg"):
            key = os.path.splitext(f)[0].lower()
            path = os.path.join(CAT_DIR, f)
            img = cv2.imread(path)
            if img is not None:
                cat_images[key] = img
            else:
                print(f"[WARN] Could not load {path}")
else:
    print(f"[ERROR] Cats folder not found: {CAT_DIR}")

print(f"Loaded {len(cat_images)} cat images:", list(cat_images.keys()))

cap = cv2.VideoCapture(0)

# --- Mouth state tracking ---
# states: "normal" (default), "open", "shut"
mouth_state = "normal"
last_mouth_closed_time = 0.0

# --- Main loop ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    pose_results = pose_detector.process(rgb)

    head_angle = 0.0
    left_hand_up = False
    right_hand_up = False
    mouth_open = False

    # --- Analyze face for head tilt and mouth ---
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        # head tilt (eyes)
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        head_angle = np.degrees(np.arctan2(dy, dx))

        # mouth open detection (upper/lower lip landmarks)
        # using 13 (upper lip) and 14 (lower lip) as in your baseline
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]
        mouth_dist = abs(bottom_lip.y - top_lip.y)
        mouth_open = mouth_dist > MOUTH_OPEN_THRESH

        # Update mouth_state machine only when face detected
        now = time.time()
        if mouth_open:
            # immediate switch to open
            mouth_state = "open"
            # reset closed timer
            last_mouth_closed_time = 0.0
        else:
            # mouth is currently shut
            if mouth_state == "open":
                # just transitioned from open -> shut
                mouth_state = "shut"
                last_mouth_closed_time = now
            elif mouth_state == "shut":
                # still shut: check timeout to revert to normal
                if last_mouth_closed_time == 0.0:
                    last_mouth_closed_time = now
                elif now - last_mouth_closed_time >= MOUTH_SHUT_TIMEOUT:
                    mouth_state = "normal"
                    last_mouth_closed_time = 0.0
            # if mouth_state already "normal", remain normal

    # --- Analyze pose for hand raises ---
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark

        # safe-get helper for landmarks
        def get_landmark(name):
            try:
                return pose_landmarks[getattr(mp_pose.PoseLandmark, name)]
            except Exception:
                return None

        lw = get_landmark("LEFT_WRIST")
        rw = get_landmark("RIGHT_WRIST")
        le = get_landmark("LEFT_ELBOW")
        re = get_landmark("RIGHT_ELBOW")

        if lw and le:
            left_hand_up = lw.y < (le.y - ELBOW_BUFFER)
        if rw and re:
            right_hand_up = rw.y < (re.y - ELBOW_BUFFER)

    # --- Decide which cat image to show ---
    # Priority: hand raises -> head tilt -> mouth states -> default normal
    cat_img = cat_images.get("cat_normal")
    current_pose = "normal"

    # Hands first
    if left_hand_up and right_hand_up:
        cat_img = cat_images.get("cat_raise_both", cat_img)
        current_pose = "raise_both"
    elif left_hand_up:
        cat_img = cat_images.get("cat_raise_right", cat_img)
        current_pose = "raise_right"
    elif right_hand_up:
        cat_img = cat_images.get("cat_raise_left", cat_img)
        current_pose = "raise_left"
    else:
        # Head tilt (if no hand raise)
        if head_angle > 15:
            cat_img = cat_images.get("cat_right", cat_img)
            current_pose = "look_right"
        elif head_angle < -15:
            cat_img = cat_images.get("cat_left", cat_img)
            current_pose = "look_left"
        else:
            # Mouth-state handling only when no hand or head override
            if mouth_state == "open":
                cat_img = cat_images.get("talk_open", cat_img)
                current_pose = "talk_open"
            elif mouth_state == "shut":
                cat_img = cat_images.get("talk_shut", cat_img)
                current_pose = "talk_shut"
            else:
                cat_img = cat_images.get("cat_normal", cat_img)
                current_pose = "normal"

    # --- Debug overlays (optional) ---
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Overlay status text
    cv2.putText(frame, f"Head tilt: {head_angle:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Hands: L={'Up' if left_hand_up else 'Down'}, R={'Up' if right_hand_up else 'Down'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Mouth state: {mouth_state}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    cv2.putText(frame, f"Cat pose: {current_pose}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Combine with cat image ---
    if cat_img is not None:
        cat_display = cv2.resize(cat_img, (frame.shape[1], frame.shape[0]))
    else:
        cat_display = np.zeros_like(frame)
        cv2.putText(cat_display, "No cat image loaded!", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    combined = np.hstack((frame, cat_display))
    cv2.imshow("MeoMe Mirror (ESC to exit)", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
