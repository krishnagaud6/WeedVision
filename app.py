import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="WeedVision", page_icon="🌱", layout="wide")
st.title("🌱 WeedVision")
st.write("Upload an image to run weed detection with your trained model.")

@st.cache_resource
def load_model():
    # best.pt must be in the same folder as this app.py
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as exc:
    st.error("Could not load best.pt. Check that the model file and dependencies are present.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Choose a plant or field image", type=["jpg", "jpeg", "png", "webp"])
confidence = st.slider("Detection confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    if st.button("Detect weeds", type="primary"):
        with st.spinner("Running WeedVision..."):
            predictions = model.predict(source=image, conf=confidence, verbose=False)

        result = predictions[0]
        # Ultralytics plot() returns a BGR NumPy image; convert it for Streamlit.
        annotated_rgb = result.plot()[:, :, ::-1]

        with right:
            st.subheader("Detection result")
            st.image(annotated_rgb, use_container_width=True)

        if result.boxes is None or len(result.boxes) == 0:
            st.info("No objects detected above this confidence threshold.")
        else:
            rows = []
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                rows.append({
                    "Detected class": names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id],
                    "Confidence": f"{score:.1%}",
                })
            st.subheader("Detections")
            st.dataframe(rows, use_container_width=True, hide_index=True)

st.caption("Demo note: predictions depend on the model's training classes and input quality.")
