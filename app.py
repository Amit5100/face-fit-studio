"""
Face Fit Studio
---------------
Streamlit app: background photo mein face detect karta hai (MediaPipe) aur
upload kiye gaye video ko us face ke exact area par overlay karke naya MP4 banata hai (OpenCV).
UI: iOS-style glassmorphism + smooth animations.
Deploy: Hugging Face Spaces (Streamlit SDK).
"""

import os
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Face Fit Studio", page_icon="🎬", layout="centered")

MAX_BG_WIDTH = 1920  # HD quality ke liye max width


# ---------------------------------------------------------------------------
# CSS: Glassmorphism + animations
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'SF Pro Display', sans-serif; }

/* Streamlit default chrome hide */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* ---------- Animated iOS-style background ---------- */
.stApp {
    background: linear-gradient(135deg, #1b1f4b 0%, #4a1d6e 45%, #0f5c7a 100%);
    background-size: 200% 200%;
    animation: bgShift 18s ease-in-out infinite;
    color: #fff;
}
.stApp::before, .stApp::after {
    content: ""; position: fixed; border-radius: 50%; filter: blur(80px);
    z-index: 0; pointer-events: none; opacity: .55;
}
.stApp::before { width: 420px; height: 420px; background: #ff5fa2; top: -120px; left: -100px;
                 animation: floatA 14s ease-in-out infinite; }
.stApp::after  { width: 380px; height: 380px; background: #35c6ff; bottom: -120px; right: -80px;
                 animation: floatB 16s ease-in-out infinite; }

.block-container { position: relative; z-index: 1; max-width: 760px; padding-top: 2.5rem; }

/* ---------- Keyframes ---------- */
@keyframes bgShift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes floatA  { 0%,100% { transform: translate(0,0); } 50% { transform: translate(60px,80px); } }
@keyframes floatB  { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-70px,-60px); } }
@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: translateY(0); } }
@keyframes spin   { to { transform: rotate(360deg); } }
@keyframes pulse  { 0%,100% { transform: scale(.85); opacity: .5; } 50% { transform: scale(1.15); opacity: 1; } }
@keyframes bounce { 0% { transform: scale(1); } 30% { transform: scale(.90); } 60% { transform: scale(1.08); } 100% { transform: scale(1); } }

/* ---------- Glass card (hero + st.container(key="glass_*")) ---------- */
.glass, [class*="st-key-glass"] {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 28px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    animation: fadeSlideUp .8s cubic-bezier(.2,.8,.2,1) both;
    transition: transform .35s ease, box-shadow .35s ease;
}
.glass:hover, [class*="st-key-glass"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 26px 60px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.3);
}
.st-key-glass_a { animation-delay: .15s; }
.st-key-glass_b { animation-delay: .30s; }
.st-key-glass_c { animation-delay: .45s; }

.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 .3rem 0; letter-spacing: -0.02em; color: #fff; }
.hero p  { margin: 0; color: rgba(255,255,255,.75); font-weight: 300; font-size: 1.05rem; }

h3, label, p, span, .stMarkdown { color: #fff !important; }

/* ---------- File uploaders ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.08);
    border: 1.5px dashed rgba(255,255,255,.4);
    border-radius: 20px;
    transition: transform .3s ease, background .3s ease, border-color .3s ease, box-shadow .3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    transform: translateY(-3px) scale(1.015);
    background: rgba(255,255,255,.16);
    border-color: rgba(255,255,255,.85);
    box-shadow: 0 12px 30px rgba(0,0,0,.3);
}
[data-testid="stFileUploaderDropzone"] * { color: rgba(255,255,255,.9) !important; }

/* ---------- Buttons (Process / Play / Download) ---------- */
.stButton > button, .stDownloadButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(255,255,255,.28), rgba(255,255,255,.1));
    color: #fff !important;
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 18px;
    padding: .8rem 1.2rem;
    font-weight: 600; font-size: 1.05rem;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 10px 25px rgba(0,0,0,.3);
    transition: transform .25s cubic-bezier(.3,1.4,.5,1), box-shadow .25s ease, background .25s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-4px) scale(1.04);
    background: linear-gradient(135deg, rgba(255,255,255,.4), rgba(255,255,255,.18));
    box-shadow: 0 16px 34px rgba(0,0,0,.4);
    border-color: rgba(255,255,255,.7);
}
.stButton > button:active, .stDownloadButton > button:active { animation: bounce .4s ease; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid #fff; outline-offset: 3px; }

/* ---------- Processing indicator ---------- */
.loader-wrap { display: flex; flex-direction: column; align-items: center; gap: 1rem; padding: 1.2rem 0; }
.ring { width: 64px; height: 64px; border-radius: 50%;
        border: 5px solid rgba(255,255,255,.2); border-top-color: #fff;
        animation: spin 1s linear infinite; box-shadow: 0 0 25px rgba(255,255,255,.35); }
.dots { display: flex; gap: 8px; }
.dots span { width: 10px; height: 10px; border-radius: 50%; background: #fff; animation: pulse 1.2s ease-in-out infinite; }
.dots span:nth-child(2) { animation-delay: .2s; } .dots span:nth-child(3) { animation-delay: .4s; }
.loader-text { color: rgba(255,255,255,.85); font-weight: 400; }

/* Progress bar, video, images */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #ff5fa2, #35c6ff); }
video, [data-testid="stImage"] img { border-radius: 20px; box-shadow: 0 12px 35px rgba(0,0,0,.4); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def loader_html(text: str) -> str:
    """Animated spinner + pulsating dots ka HTML."""
    return (
        '<div class="loader-wrap"><div class="ring"></div>'
        '<div class="dots"><span></span><span></span><span></span></div>'
        f'<div class="loader-text">{text}</div></div>'
    )


def load_background(file) -> np.ndarray:
    """Uploaded photo ko BGR numpy array mein decode karo (even dimensions ke saath)."""
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Background photo read nahi ho payi. Dusri JPG/PNG try karein.")
    h, w = img.shape[:2]
    if w > MAX_BG_WIDTH:
        scale = MAX_BG_WIDTH / w
        img = cv2.resize(img, (MAX_BG_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]  # H.264 ke liye even size


def detect_face(img: np.ndarray, padding: float):
    """MediaPipe se sabse bada face dhundho. Return: (x, y, w, h) pixels mein ya None."""
    H, W = img.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        result = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.detections:
        return None

    def area(d):
        b = d.location_data.relative_bounding_box
        return b.width * b.height

    bb = max(result.detections, key=area).location_data.relative_bounding_box
    x, y = bb.xmin * W, bb.ymin * H
    w, h = bb.width * W, bb.height * H

    # Padding: box ko thoda bada karo (forehead/chin cover karne ke liye)
    x -= w * padding / 2
    y -= h * padding / 2
    w *= 1 + padding
    h *= 1 + padding

    # Image ke andar clamp karo
    x, y = max(0, int(x)), max(0, int(y))
    w, h = min(int(w), W - x), min(int(h), H - y)
    return (x, y, w, h) if w > 10 and h > 10 else None


def cover_resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Frame ko (w, h) mein fit karo bina stretch kiye: scale + center crop."""
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    nw, nh = max(w, int(fw * scale)), max(h, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def build_video(bg, box, src_path, out_path, on_progress):
    """Har video frame ko face area par paste karke final HD MP4 (H.264 + Original FPS + Audio) banao."""
    x, y, w, h = box
    H, W = bg.shape[:2]

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError("Video open nahi ho paya. MP4/MOV file check karein.")
    
    # Original video ka exact FPS detect karo taaki video slow ya fast/stutter na ho
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 120:
        fps = 30.0  # Default standard smooth FPS
        
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    raw_path = out_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = bg.copy()
        canvas[y : y + h, x : x + w] = cover_resize(frame, w, h)
        writer.write(canvas)
        i += 1
        if i % 5 == 0:
            on_progress(min(i / total, 1.0) * 0.85)
    cap.release()
    writer.release()
    if i == 0:
        raise ValueError("Video mein koi frame nahi mila.")

    # Chrome-playable H.264 + Original Audio + Same FPS ke sath FFmpeg re-encode
    on_progress(0.9)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path, "-i", src_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-r", str(fps), "-preset", "medium", "-crf", "18", 
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        os.replace(raw_path, out_path)  # fallback: bina re-encode
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    on_progress(1.0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="glass hero"><h1>🎬 Face Fit Studio</h1>'
    "<p>Photo ke face par apna video automatically fit karein.</p></div>",
    unsafe_allow_html=True,
)

with st.container(key="glass_a"):
    st.markdown("### 📸 Background photo")
    photo = st.file_uploader("JPG ya PNG", type=["jpg", "jpeg", "png"], key="photo")
    if photo:
        st.image(photo, use_container_width=True)

with st.container(key="glass_b"):
    st.markdown("### 🎞️ Face video")
    video = st.file_uploader("MP4 ya MOV", type=["mp4", "mov"], key="video")
    padding = st.slider("Face area padding (%)", 0, 40, 10, help="Detect kiye box ko bada karta hai") / 100
    process = st.button("✨ Process", disabled=not (photo and video))

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if process and photo and video:
    status = st.empty()
    bar = st.progress(0.0)
    status.markdown(f'<div class="glass">{loader_html("Face detect ho raha hai...")}</div>', unsafe_allow_html=True)

    tmp_dir = tempfile.mkdtemp()
    try:
        bg = load_background(photo)
        box = detect_face(bg, padding)
        if box is None:
            st.error("Photo mein koi face nahi mila. Clear front-facing photo try karein.")
            st.session_state.pop("result", None)
        else:
            src_path = os.path.join(tmp_dir, "input" + os.path.splitext(video.name)[1])
            with open(src_path, "wb") as f:
                f.write(video.getvalue())
            out_path = os.path.join(tmp_dir, "output.mp4")

            def on_progress(p):
                bar.progress(p)
                status.markdown(
                    f'<div class="glass">{loader_html(f"Video process ho rahi hai... {int(p * 100)}%")}</div>',
                    unsafe_allow_html=True,
                )

            build_video(bg, box, src_path, out_path, on_progress)
            
            # Save the file path in session state instead of raw bytes
            st.session_state["result"] = out_path
            st.session_state["show_video"] = False
    except Exception as e:  # user-friendly error
        st.error(f"Error: {e}")
        st.session_state.pop("result", None)
    finally:
        status.empty()
        bar.empty()

# ---------------------------------------------------------------------------
# Result: Play + Download
# ---------------------------------------------------------------------------
if st.session_state.get("result") and os.path.exists(st.session_state["result"]):
    with st.container(key="glass_c"):
        st.markdown("### ✅ Aapka video taiyaar hai")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Play Video", key="play"):
                st.session_state["show_video"] = True
        with c2:
            with open(st.session_state["result"], "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "⬇️ Download Video",
                data=video_bytes,
                file_name="face_fit_video.mp4",
                mime="video/mp4",
            )
        if st.session_state.get("show_video"):
            st.video(st.session_state["result"], format="video/mp4")
```[cite: 10, 11]