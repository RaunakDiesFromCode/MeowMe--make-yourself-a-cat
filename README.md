# 🐱 MeoMe Mirror

A fun interactive mirror app that mimics your facial expressions and gestures with cat meme images using **OpenCV** and **MediaPipe**.

---

## 📋 Features

* **Head tilt tracking:**
  Tilt your head left or right — the cat looks the same way.

* **Hand raise detection:**
  Raise your **left**, **right**, or **both** hands (up to the elbow level or slightly higher) — the cat mirrors it.

* **Mouth detection (talking):**
  When you start talking, the cat switches between **talk open** and **talk shut**.
  If you stop talking for more than 1 second, it goes back to normal, else it keeps switching.

* **Smile detection:**
  If you smile, the cat replaces its **normal** pose with a **smile** (if `cat_smile.jpg` exists).

---

## 🐾 Required Cat Images

Your `/cats` folder should contain:

```
cats/
│
├── cat_normal.jpg
├── cat_left.jpg
├── cat_right.jpg
├── cat_raise_left.jpg
├── cat_raise_right.jpg
├── cat_raise_both.jpg
├── talk_open.jpg
├── talk_shut.jpg
└── cat_smile.jpg   (optional)
```

> 🖍️ If any image is missing, the program will automatically fall back to `cat_normal.jpg`.

---

## ⚙️ Installation

### 1️⃣ Clone or download the project:

```bash
git clone https://github.com/yourusername/meome-mirror.git
cd meome-mirror
```

### 2️⃣ Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

### 3️⃣ Add your `cats/` folder:

Place all your cat `.jpg` files in a folder named **cats** beside the script.

---

## ▶️ Usage

Run the script:

```bash
python cat_mirror.py
```

A window opens showing:

* Your webcam feed (left side) with debug lines/points
* The matching cat image (right side)

Press **ESC** to exit.

---

## 🧠 How It Works

| Detection      | Method                                           | Trigger         |
| -------------- | ------------------------------------------------ | --------------- |
| **Head Tilt**  | Angle between left and right eyes                | > ±15°          |
| **Hand Raise** | Wrist Y position higher than elbow (with buffer) | Left/Right/Both |
| **Talking**    | Distance between upper and lower lips            | > 0.02          |
| **Smile**      | Width between mouth corners                      | > 0.045         |

The program uses **MediaPipe FaceMesh** for facial landmarks and **MediaPipe Pose** for body keypoints.

---

## 🧪 Notes

* Adjust thresholds (`MOUTH_OPEN_THRESH`, `SMILE_THRESH`, etc.) for your face distance.
* Requires a decent webcam and lighting for best detection.
* Works cross-platform (Windows/macOS/Linux).

---

## 🐈 Example Interaction

| You Do           | Cat Reaction                                    |
| ---------------- | ----------------------------------------------- |
| Look Left        | 🐱 → `cat_left.jpg`                             |
| Look Right       | 🐱 → `cat_right.jpg`                            |
| Raise Right Hand | 🐱 → `cat_raise_right.jpg`                      |
| Talk             | 🐱 → switches between `talk_open` & `talk_shut` |
| Smile            | 🐱 → `cat_smile.jpg`                            |

---

## 🦩 Credits

* [MediaPipe](https://github.com/google/mediapipe) for landmark tracking.
* [OpenCV](https://opencv.org/) for image processing.
