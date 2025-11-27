# app_streamlit.py
import os
import tempfile
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import cv2  # for video reading/display (opencv-python)

# ---------- Helpers & small wrappers ----------
st.set_page_config(layout="wide", page_title="Video Model Demo")

@st.cache_resource
def load_tokenizer_and_model(model_path: str, hf_token: str | None = None):
    """Minimal model loader — adapt the model class to your model type."""
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
        kwargs["use_auth_token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **kwargs)
    # Choose appropriate class for your model. Replace with the correct one.
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.eval()
    return tokenizer, model

def load_pkl(path: str) -> Any:
    """Load pickle safely — ensures numpy available and returns object."""
    with open(path, "rb") as f:
        return pickle.load(f)

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temp file and return path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def read_video_frames(video_path: str, max_frames: int = 120):
    """Simple generator to read frames using cv2. Yields (frame_index, frame_bgr)."""
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened() and idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1
    cap.release()

# ---------- PLACEHOLDER: your notebook's inference function ----------
def run_inference_on_video(video_path: str, model, tokenizer) -> Dict:
    """
    Replace/adapt this function with your notebook code.
    It should:
      - open the video (path)
      - run model inference (per frame / per clip)
      - return a dictionary with results, e.g. {"predictions": [...], "metadata": {...}}
    Example stub below returns dummy values.
    """
    # ---- BEGIN SAMPLE STUB (replace with your real code) ----
    outs = {"predictions": []}
    for idx, frame in read_video_frames(video_path, max_frames=30):
        # pretend we convert frame -> text -> model -> label
        # in reality, do resizing, transform to tensors and call model(...)
        outs["predictions"].append({"frame": int(idx), "label": "unknown", "score": 0.0})
    outs["metadata"] = {"frames_processed": len(outs["predictions"])}
    return outs
    # ---- END SAMPLE STUB ----

# ---------- Streamlit UI ----------
st.title("Video Model — Streamlit Demo")
st.markdown("Upload a video or choose a local video path, load model, then run inference.")

with st.sidebar:
    st.header("Model & artifacts")
    model_path = st.text_input("Model folder or HF repo id", value="")
    hf_token = st.text_input("Hugging Face token (optional)", type="password", value=st.secrets.get("HF_TOKEN", ""))
    artifact_pkl = st.file_uploader("Optional: upload .pkl artifact (embeddings / metadata)", type=["pkl"])
    upload_local_video = st.file_uploader("Upload video (mp4 / mov)", type=["mp4", "mov", "avi"])
    local_video_path = st.text_input("Or local video path (leave blank if uploading)", value="")
    max_frames = st.slider("Max frames to preview/infer (for speed)", min_value=1, max_value=300, value=60)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model loader")
    if st.button("Load model"):
        if not model_path:
            st.error("Enter model_path (local folder or HF repo id)")
        else:
            try:
                with st.spinner("Loading tokenizer and model..."):
                    tokenizer, model = load_tokenizer_and_model(model_path, hf_token if hf_token else None)
                st.session_state["model_loaded"] = True
                st.session_state["tokenizer"] = tokenizer
                st.session_state["model"] = model
                st.success("Model loaded.")
                st.write("Tokenizer:", type(tokenizer).__name__)
                st.write("Model:", type(model).__name__)
            except Exception as e:
                st.session_state["model_loaded"] = False
                st.error("Model load failed. See details below.")
                st.text(traceback.format_exc())

    st.markdown("---")
    st.subheader("Artifacts")
    if artifact_pkl:
        try:
            pkl_path = save_uploaded_file(artifact_pkl)
            obj = load_pkl(pkl_path)
            st.success("Pickle loaded.")
            st.write("Type:", type(obj))
            # show small sample
            if isinstance(obj, dict):
                st.json({k: str(v)[:400] for k, v in list(obj.items())[:10]})
            elif isinstance(obj, (list, tuple)):
                st.write("Length:", len(obj))
                st.write("First items:", obj[:5])
            else:
                st.write(str(obj)[:400])
            st.session_state["artifact"] = obj
        except Exception as e:
            st.error("Failed to load .pkl: ensure dependencies (numpy) are installed.")
            st.text(traceback.format_exc())

with col2:
    st.subheader("Video input")
    video_path_to_use = None
    if upload_local_video:
        saved = save_uploaded_file(upload_local_video)
        st.video(saved)
        video_path_to_use = saved
    elif local_video_path:
        if os.path.exists(local_video_path):
            st.video(local_video_path)
            video_path_to_use = local_video_path
        else:
            st.info("Local path not found — upload a file instead or check path.")
    else:
        st.info("Upload a video or set a local path to run inference.")

    st.markdown("---")
    if st.button("Run inference"):
        if not st.session_state.get("model_loaded", False):
            st.error("Please load the model first from the sidebar.")
        elif not video_path_to_use:
            st.error("Please provide a video via upload or local path.")
        else:
            try:
                with st.spinner("Running inference (this may take a while)..."):
                    tokenizer = st.session_state["tokenizer"]
                    model = st.session_state["model"]
                    results = run_inference_on_video(video_path_to_use, model, tokenizer)
                st.success("Inference completed.")
                # show summary
                st.json({"frames_processed": results.get("metadata", {}).get("frames_processed", None)})
                # show first N predictions
                preds = results.get("predictions", [])
                if preds:
                    st.write("First predictions:")
                    st.table(preds[:10])
                # allow download of results
                out_json = json.dumps(results, indent=2)
                st.download_button("Download results (JSON)", data=out_json, file_name="results.json", mime="application/json")
            except Exception:
                st.error("Inference failed — see traceback.")
                st.text(traceback.format_exc())

st.markdown("---")
st.markdown("### Developer notes")
st.markdown(
    "- Edit `run_inference_on_video()` to port the code from your notebook (`videoinput_pretrained.ipynb`).\n"
    "- Use smaller `max_frames` when debugging to save time.\n"
    "- If your notebook uses TensorFlow, PyTorch, or NumPy, ensure those packages are in `requirements.txt`.\n"
)


