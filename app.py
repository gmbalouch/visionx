import json
import os
import time
import cv2
import numpy as np
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

def get_secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return None

def build_rtc_config():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    user, pwd = get_secret("TURN_USERNAME"), get_secret("TURN_PASSWORD")
    if user and pwd:
        servers += [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
            {"urls": ["turn:global.relay.metered.ca:80",
                      "turn:global.relay.metered.ca:80?transport=tcp",
                      "turn:global.relay.metered.ca:443",
                      "turns:global.relay.metered.ca:443?transport=tcp"],
             "username": user, "credential": pwd},
        ]
    return RTCConfiguration({"iceServers": servers})

RTC_CONFIG = build_rtc_config()


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
    else ["Live (browser camera)", "Inspect snapshot", "About"]
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
    cam = st.radio("Camera", ["Front camera", "Back camera"], horizontal=True)
    st.caption("To switch cameras: press STOP first, change the option, then press START.")
    facing = "user" if cam == "Front camera" else "environment"
    facing_constraint = {"ideal": "user"} if cam == "Front camera" else {"exact": "environment"}

    def video_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        result = model(img, conf=conf, imgsz=320, verbose=False)[0]
        return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

    webrtc_streamer(
        key=f"visionx-live-{facing}",
        video_frame_callback=video_callback,
        rtc_configuration=RTC_CONFIG,
        server_rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={
            "video": {"facingMode": facing_constraint},
            "audio": False,
        },
        async_processing=True,
    )

# ---------- ABOUT PAGE ----------
elif mode == "About":
    # EDIT THESE: replace with your real names and roles
    TEAM = [
        ("Ghulam Mustafa", "Software Engineering / 23SW086"),
        ("Member 2", "Team member / role"),
        ("Member 3", "Team member / role"),
        ("Member 3", "Team member / role"),
    ]

    st.header("About VisionX")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Overview", "How it works", "Dataset", "Team and internship", "Limits and future work"]
    )

    with t1:
        st.subheader("Problem")
        st.write(
            "Most computer vision demos only draw boxes around objects. They do not tell "
            "the user anything useful about what the camera sees. VisionX combines "
            "real-time object detection with an interactive information layer."
        )
        st.subheader("What it does")
        st.markdown(
            "- Detects objects live through a webcam or phone camera (front and back)\n"
            "- Shows bounding boxes, labels and confidence scores\n"
            "- Lets the user select a detected object and read its category and description\n"
            "- Generates a short AI explanation of the selected object (Gemini API)\n"
            "- Runs as a web app, so it works on laptops and phones through a link"
        )

    with t2:
        st.subheader("Pipeline")
        st.markdown(
            "1. **Camera** (browser camera via WebRTC, or OpenCV locally)\n"
            "2. **YOLO11 nano** detects objects in each frame\n"
            "3. **Post-processing** keeps boxes above the confidence threshold\n"
            "4. **Knowledge base** (JSON) maps each class to a category and description\n"
            "5. **AI explanation** (Gemini, with model fallback and caching)\n"
            "6. **Streamlit interface** shows the video, results and information"
        )
        st.subheader("Technology")
        st.table({
            "Part": ["Language", "Detection", "Vision", "Interface", "Live video", "AI text", "Hosting"],
            "Tool": ["Python", "Ultralytics YOLO11 (PyTorch)", "OpenCV, NumPy", "Streamlit",
                     "streamlit-webrtc (with a TURN relay)", "Google Gemini API",
                     "Streamlit Community Cloud + GitHub"],
        })

    with t3:
        st.subheader("COCO dataset")
        st.write(
            "The pre-trained model was trained on the COCO dataset (Common Objects in "
            "Context, created by Microsoft). Its detection task has 80 object classes. "
            "The model can only recognise these classes, so anything else gets mapped to "
            "the closest known class."
        )
        names = model.names
        c1, c2, c3 = st.columns(3)
        c1.metric("Classes in the model", len(names))
        c2.metric("Entries in our knowledge base", sum(1 for n in names.values() if n in kb))
        c3.metric("Model in use", model_name)
        st.dataframe(
            {
                "ID": list(names.keys()),
                "Class": list(names.values()),
                "Category": [get_info(n)["category"] for n in names.values()],
                "Description": [get_info(n)["description"] for n in names.values()],
            },
            use_container_width=True,
            height=2900,
        )

    with t4:
        st.subheader("Internship")
        st.write(
            "VisionX was developed as the project for the Catch VR one-month internship, "
            "where each team studied and extended an AI or VR open-source project. "
            "We studied Ultralytics YOLO and built an interactive assistant on top of it."
        )
        st.subheader("Team")
        st.table({"Name": [m[0] for m in TEAM], "Department/Rollno": [m[1] for m in TEAM]})
        st.caption("Built on the open-source Ultralytics YOLO project: github.com/ultralytics/ultralytics")

    with t5:
        st.subheader("Limitations")
        st.markdown(
            "- Only 80 COCO classes: unknown objects (for example glasses or pens) are "
            "matched to the closest known class and can be wrong\n"
            "- The nano model is fast but less accurate than larger models\n"
            "- Free hosting runs on CPU, so online live video has a low frame rate\n"
            "- The AI explanation can contain mistakes and needs internet"
        )
        st.subheader("Future work")
        st.markdown(
            "- Open-vocabulary detection (YOLO-World) or custom training for new objects\n"
            "- Native mobile app (on-device model export)\n"
            "- AR/VR display of detected objects and their information (Unity or A-Frame)\n"
            "- Multi-language descriptions and voice output for accessibility"
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