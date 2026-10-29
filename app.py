import json
import os
import time
import cv2
import numpy as np
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
LOCAL = os.environ.get("VISIONX_LOCAL") == "1"

st.set_page_config(page_title="VisionX", layout="wide")
st.title("VisionX: AI Visual Assistant")

# ---------- Loaders ----------
@st.cache_resource
def load_model(name):
    return YOLO(name)

@st.cache_data
def load_kb():
    with open("knowledge.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def _ai_cache():
    return {}

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

def get_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def explain_with_ai(name):
    cache = _ai_cache()
    if name in cache:
        return cache[name]
    key = get_key()
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
    except Exception as e:
        print("Gemini setup error:", e)
        return None
    prompt = f"Explain what a '{name}' is in 2 simple sentences for a student."
    for model_id in GEMINI_MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(model=model_id, contents=prompt)
                cache[name] = resp.text
                return resp.text
            except Exception as e:
                print(f"{model_id} failed: {e}")
                msg = str(e)
                if "503" in msg or "429" in msg or "UNAVAILABLE" in msg:
                    time.sleep(1.5)
                    continue
                break
    return None

# ---------- Sidebar ----------
model_name = st.sidebar.selectbox("Model", ["yolo11n.pt", "yolo11s.pt"] if LOCAL else ["yolo11n.pt"])
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
modes = ["Live detection", "Live (browser camera)", "Inspect snapshot"] if LOCAL \
    else ["Live (browser camera)", "Inspect snapshot"]
mode = st.sidebar.radio("Mode", modes)

model = load_model(model_name)
kb = load_kb()

def get_info(name):
    return kb.get(name, {"category": "Unknown",
                         "description": "No information in the knowledge base yet."})

def detect(frame_bgr):
    result = model(frame_bgr, conf=conf, verbose=False)[0]
    dets = [(model.names[int(c)], float(p))
            for c, p in zip(result.boxes.cls, result.boxes.conf)]
    return result.plot(), dets

# ---------- MODE 1: live camera (local laptop only) ----------
if mode == "Live detection":
    run = st.checkbox("Start camera")
    col1, col2 = st.columns([3, 1])
    frame_box = col1.empty()
    info_box = col2.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            start = time.time()
            annotated, dets = detect(frame)
            fps = 1 / max(time.time() - start, 1e-6)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_box.image(annotated[:, :, ::-1], channels="RGB")
            seen = sorted({n for n, _ in dets})
            text = "### Detected\n" + "\n".join(
                f"- **{n}** ({get_info(n)['category']}): {get_info(n)['description']}"
                for n in seen)
            info_box.markdown(text if seen else "### Detected\nNothing yet")
        cap.release()

# ---------- MODE 2: live camera through the browser (works online) ----------
elif mode == "Live (browser camera)":
    def video_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        result = model(img, conf=conf, imgsz=320, verbose=False)[0]
        return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

    webrtc_streamer(
        key="visionx-live",
        video_frame_callback=video_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ---------- MODE 3: snapshot + select + info ----------
else:
    img = st.camera_input("Take a picture")
    if img:
        frame = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        annotated, dets = detect(frame)

        col1, col2 = st.columns([3, 2])
        col1.image(annotated[:, :, ::-1], channels="RGB")

        with col2:
            if not dets:
                st.warning("No objects detected. Try better lighting or a lower threshold.")
            else:
                labels = [f"{n} ({c:.0%})" for n, c in dets]
                choice = st.selectbox("Select a detected object", labels)
                name = dets[labels.index(choice)][0]
                info = get_info(name)
                st.subheader(name.title())
                st.caption(f"Category: {info['category']}")
                st.write(info["description"])

                if st.button("Explain with AI"):
                    with st.spinner("Thinking..."):
                        answer = explain_with_ai(name)
                    if answer:
                        st.info(answer)
                    else:
                        st.warning("AI is busy right now. Showing the built-in description instead.")
                        st.info(info["description"])