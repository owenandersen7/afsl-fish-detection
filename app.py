import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================
# Konfigurasi Awal
# =========================
st.set_page_config(page_title="AFSL - Fish Detection", layout="wide")
st.title("🐟 AFSL - Fish Detection Webapp (Mode Cepat Default)")

MODEL_PATH = "models/best.pt"

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}")
    st.stop()

st.write("📦 Memuat model YOLOv11s...")
model = YOLO(MODEL_PATH)
st.success("Model berhasil dimuat!")

# =========================
# Parameter Mode Cepat
# =========================
SKIP_FRAME_UPLOAD = 3   # Lewati 2 frame, proses 1 frame (video upload)
SKIP_FRAME_RTC = 5      # Lewati 4 frame, proses 1 frame (real-time)
RESIZE_WIDTH = 640      # Lebar frame untuk tampilan
IMGSZ = 320             # Resolusi input model

# =========================
# Pilihan Mode
# =========================
mode = st.radio(
    "Pilih Mode Aplikasi:",
    ["Upload Video", "Kamera Real-Time"]
)

# =========================
# Fungsi Pemrosesan Video Upload
# =========================
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frame
        if frame_count % SKIP_FRAME_UPLOAD != 0:
            frame_count += 1
            continue

        # Resize frame
        height = int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1]))
        frame = cv2.resize(frame, (RESIZE_WIDTH, height))

        # Deteksi YOLO
        results = model(frame, imgsz=IMGSZ)
        annotated_frame = results[0].plot()

        # Tampilkan frame
        stframe.image(annotated_frame, channels="BGR")
        frame_count += 1

    cap.release()

# =========================
# Mode Upload Video
# =========================
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload video ikan", type=["mp4", "avi", "mov"])
    if uploaded_file:
        st.write("📂 File diterima:", uploaded_file.name)
        with st.spinner("Memproses video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()  # Tutup file sebelum dipakai OpenCV
            process_video(tfile.name, model)
            os.unlink(tfile.name)  # Hapus setelah selesai
        st.success("✅ Selesai memproses video!")

# =========================
# Mode Kamera Real-Time
# =========================
elif mode == "Kamera Real-Time":
    st.info("📸 Arahkan kamera ke akuarium atau sumber video ikan.")

    class YOLOTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0

        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Skip frame lebih agresif untuk real-time
            if self.frame_count % SKIP_FRAME_RTC != 0:
                return img

            # Resize frame
            height = int(img.shape[0] * (RESIZE_WIDTH / img.shape[1]))
            img = cv2.resize(img, (RESIZE_WIDTH, height))

            # Deteksi YOLO
            results = model(img, imgsz=IMGSZ)
            return results[0].plot()

    webrtc_streamer(
        key="fish-detection",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},  # resolusi rendah untuk HP
            "audio": False
        }
    )
