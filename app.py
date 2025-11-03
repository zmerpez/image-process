import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("YOLO Object Detection App")

# ------------------------------------
# Load YOLO model (auto-downloads weights if missing)
# ------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8s.pt")  # small, fast model
    return model

model = load_model()


# ------------------------------------
# Details section
# ------------------------------------
st.markdown("---")
st.subheader("Details & References")

st.markdown("""
**What is YOLO?**  
YOLO (You Only Look Once) is a real-time object detection algorithm that predicts bounding boxes and class probabilities directly from full images in a single evaluation. It is fast and accurate, making it popular for applications like video analysis, robotics, and self-driving cars.

**How this app works:**  
1. You upload an image or provide an image URL.  
2. YOLOv8 predicts objects in the image and draws bounding boxes with labels.  
3. The app also crops detected objects and displays them separately for clarity.  
4. Confidence threshold slider lets you control the minimum detection confidence.

**References:**  
- YOLOv8 Documentation: [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)  
- YOLO Paper: [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)  
- Ultralytics GitHub: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
""")


# YOLO Training Examples

st.markdown("---")
st.subheader("YOLO Training Examples")

html_code = """
<div style="background-color:transparent; padding:20px; border-radius:10px;">
<h4>1️⃣ Object Detection (Custom Dataset)</h4>
<p>Train YOLOv8 to detect your own objects:</p>

<h4>2️⃣ Image Classification</h4>
<p>You can use YOLOv8 for classification instead of detection:</p>


<h4>3️⃣ Regression Tasks (e.g., Predict Bounding Box Coordinates)</h4>
<p>YOLO can also be fine-tuned to regress custom numerical values (like keypoints or continuous outputs):</p>

<p><strong>References:</strong></p>
<ul>
<li><a href="https://docs.ultralytics.com/tasks/detect/">YOLOv8 Detection Training</a></li>
<li><a href="https://docs.ultralytics.com/tasks/classify/">YOLOv8 Classification Training</a></li>
<li><a href="https://docs.ultralytics.com/tutorials/custom-training/">Custom YOLO Training Tutorial</a></li>
</ul>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)

st.markdown("---")
# ------------------------------------
# Input section: Upload or URL
# ------------------------------------
st.subheader("Choose Input Method")

option = st.radio("Select input source:", ["Upload an image", "Use an image URL"])
image = None

if option == "Upload an image":
    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use an image URL":
    url = st.text_input(
        "Enter an image URL",
        "https://ultralytics.com/images/zidane.jpg"
    )
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not load image from URL: {e}")

# ------------------------------------
# Display input image
# ------------------------------------
if image is not None:
    st.image(image, caption="Input Image", width=400)
else:
    st.info("Upload an image or paste a valid URL to start detection.")
    st.stop()

# ------------------------------------
# Confidence threshold
# ------------------------------------
conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)

# ------------------------------------
# Run YOLO
# ------------------------------------
if st.button("Detect Objects"):
    with st.spinner("Running YOLO model..."):
        image_np = np.array(image)
        results = model.predict(image_np, conf=conf_threshold, verbose=False)
        res = results[0]
        annotated = res.plot()  # Annotated frame with boxes/labels

    # Show annotated image (smaller for layout)
    st.subheader("Detected Objects Overview")
    st.image(annotated, caption="YOLO Detection Results", width=600)

    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        st.success(f"Detected {len(boxes)} objects")

        # ------------------------------------
        # Show detected objects + cropped images
        # ------------------------------------
        st.subheader("Cropped Detections")

        cols = st.columns(min(4, len(boxes)))  # up to 4 per row
        image_np = np.array(image)

        for i, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
            x1, y1, x2, y2 = map(int, xyxy)
            crop = image_np[y1:y2, x1:x2]
            crop_img = Image.fromarray(crop)
            label = f"{model.names[int(cls)]} ({conf:.2f})"

            col = cols[i % len(cols)]
            col.image(crop_img, caption=label, width=150)

        st.caption("Each crop corresponds to a detected object.")
    else:
        st.warning("No objects detected. Try lowering the confidence threshold or a different image.")


