import os
import cv2
import dlib
import numpy as np
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename

# Khởi tạo Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load mô hình
haar_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
hog_detector = dlib.get_frontal_face_detector()
dnn_net = cv2.dnn.readNetFromTensorflow(
    "models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt"
)

def detect_faces(image_path, method):
    """Nhận diện khuôn mặt với phương pháp được chọn"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "haar":
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    elif method == "hog":
        faces = hog_detector(gray)
        faces = [(d.left(), d.top(), d.width(), d.height()) for d in faces]
    elif method == "dnn":
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        faces = []
        h, w = image.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))

    # Vẽ hình chữ nhật quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    return result_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return "No image uploaded", 400
    file = request.files["image"]
    method = request.form.get("method")

    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Xử lý ảnh với phương pháp nhận diện khuôn mặt
    result_path = detect_faces(file_path, method)
    return send_file(result_path, mimetype="image/jpeg")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
