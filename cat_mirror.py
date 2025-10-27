import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Settings (tweak if needed) ---
MOUTH_OPEN_THRESH = 0.02   # mouth open threshold
SMILE_THRESH = 0.065        # mouth corner distance threshold for smile
ELBOW_BUFFER = 0.04        # how much higher wrist must be than elbow
MOUTH_SHUT_TIMEOUT = 1.0   # seconds before reverting to normal after shut

# --- Init Mediapipe ---
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
pose_detector = mp_pose.Pose()

# --- Load cat images ---
CAT_DIR = os.path.join(os.path.dirname(__file__), "cats")
cat_images = {}
if os.path.exists(CAT_DIR):
    for f in os.listdir(CAT_DIR):
        if f.lower().endswith(".jpg"):
            key = os.path.splitext(f)[0].lower()
            img = cv2.imread(os.path.join(CAT_DIR, f))
            if img is not None:
                cat_images[key] = img
            else:
                print(f"[WARN] Could not load {f}")
else:
    print(f"[ERROR] Cats folder not found: {CAT_DIR}")

print(f"Loaded {len(cat_images)} cat images:", list(cat_images.keys()))

cap = cv2.VideoCapture(0)

# --- Mouth/Smile tracking ---
mouth_state = "normal"
last_mouth_closed_time = 0.0
smiling = False

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
    smiling = False  # reset each frame

    # --- FACE DETECTION ---
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        # head tilt (eyes)
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        head_angle = np.degrees(np.arctan2(dy, dx))

        # mouth open detection
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]
        mouth_dist = abs(bottom_lip.y - top_lip.y)
        mouth_open = mouth_dist > MOUTH_OPEN_THRESH

        # smile detection (distance between mouth corners)
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        mouth_width = abs(right_mouth.x - left_mouth.x)
        smiling = mouth_width > SMILE_THRESH

        # --- mouth state machine ---
        now = time.time()
        if mouth_open:
            mouth_state = "open"
            last_mouth_closed_time = 0.0
        else:
            if mouth_state == "open":
                mouth_state = "shut"
                last_mouth_closed_time = now
            elif mouth_state == "shut":
                if last_mouth_closed_time == 0.0:
                    last_mouth_closed_time = now
                elif now - last_mouth_closed_time >= MOUTH_SHUT_TIMEOUT:
                    mouth_state = "normal"
                    last_mouth_closed_time = 0.0

    # --- POSE DETECTION ---
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark
        lw = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        le = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        re = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_hand_up = lw.y < (le.y - ELBOW_BUFFER)
        right_hand_up = rw.y < (re.y - ELBOW_BUFFER)

    # --- CHOOSE CAT IMAGE ---
    cat_img = cat_images.get("cat_normal")
    current_pose = "normal"

    # Priority: hand -> head tilt -> mouth -> normal/smile
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
        # Head tilt
        if head_angle > 15:
            cat_img = cat_images.get("cat_right", cat_img)
            current_pose = "look_right"
        elif head_angle < -15:
            cat_img = cat_images.get("cat_left", cat_img)
            current_pose = "look_left"
        else:
            # Mouth animation
            if mouth_state == "open":
                cat_img = cat_images.get("talk_open", cat_img)
                current_pose = "talk_open"
            elif mouth_state == "shut":
                cat_img = cat_images.get("talk_shut", cat_img)
                current_pose = "talk_shut"
            else:
                # Normal or Smile only here
                if smiling and "cat_smile" in cat_images:
                    cat_img = cat_images["cat_smile"]
                    current_pose = "smile"
                else:
                    cat_img = cat_images.get("cat_normal", cat_img)
                    current_pose = "normal"

    # --- Draw debug info ---
    cv2.putText(frame, f"Head tilt: {head_angle:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Hands: L={'Up' if left_hand_up else 'Down'}, R={'Up' if right_hand_up else 'Down'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Mouth: {mouth_state}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    cv2.putText(frame, f"Smile: {'Yes' if smiling else 'No'}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Cat pose: {current_pose}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Display side-by-side ---
    if cat_img is not None:
        cat_display = cv2.resize(cat_img, (frame.shape[1], frame.shape[0]))
    else:
        cat_display = np.zeros_like(frame)
        cv2.putText(cat_display, "No cat image loaded!", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    combined = np.hstack((frame, cat_display))
    cv2.imshow("MeoMe Mirror (ESC to exit)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
