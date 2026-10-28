import sys
print("PYTHON USED:", sys.executable)

import json
import os
import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

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

@st.cache_data(show_spinner=False)
def explain_with_ai(name):
    """Optional stretch feature. Works only if ANTHROPIC_API_KEY is set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "AI explanation is not enabled (no API key set)."
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=200,
            messages=[{"role": "user",
                       "content": f"Explain what a '{name}' is in 2 simple sentences for a student."}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI explanation unavailable: {e}"

# ---------- Sidebar ----------
model_name = st.sidebar.selectbox("Model", ["yolo11n.pt", "yolo11s.pt"])
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
mode = st.sidebar.radio("Mode", ["Live detection", "Inspect snapshot"])

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

# ---------- MODE 1: live camera ----------
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

# ---------- MODE 2: snapshot + select + info ----------
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
                        st.info(explain_with_ai(name))