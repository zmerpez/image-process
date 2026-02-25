# Image Process

**Workshop: Getting Started with YOLO Object Detection using Streamlit**

In this hands-on workshop, participants will be introduced to **YOLO (You Only Look Once)** object detection—one of the most popular real-time computer vision models. We’ll break down how YOLO works at a high level, how it detects and classifies objects in images, and why it’s widely used in real-world applications.

To make the concepts practical and interactive, we’ll build a simple **Streamlit** web app that allows users to upload images and see YOLO detect objects in real time. Along the way, participants will learn how Streamlit can quickly turn machine learning models into clean, shareable web applications with minimal code.

By the end of the session, attendees will understand the basics of object detection, how YOLO performs inference, and how to deploy AI models through an easy-to-build Streamlit interface.
**Python 3.9–3.12** recommended.

```bash
# 1) Create/activate a virtual environment (recommended)
python -m venv .venv
# Windows: 
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
# or
pip install streamlit ultralytics pillow


# 3) Run the app
streamlit run app.py

# 4) Update requirements when installed new packages
pip freeze > requirements.txt

```


