from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the pre-trained vehicle detection model
model = tf.keras.models.load_model("car_detection_model_custom.keras")

# Folder for storing uploaded videos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    # Start vehicle detection and tracking
    return process_video(video_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Unable to read video"})

    # Resize the image and prepare it for prediction
    resized_frame = cv2.resize(frame, (128, 128))
    img_input = np.expand_dims(resized_frame, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_input)
    if prediction > 0.5:
        label = "Car Detected"
        color = (0, 0, 255)  # Red
    else:
        label = "No Car Detected"
        color = (0, 255, 0)  # Green
    
    # Vehicle tracking logic
    roi = cv2.selectROI("Select Car", frame, fromCenter=False, showCrosshair=True)
    if roi != (0, 0, 0, 0):
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, roi)

        # Create output video in MP4 format
        output_video_path = os.path.join('static', 'output_video_tracking.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update the tracker
            success, tracked_box = tracker.update(frame)
            if success:
                p1 = (int(tracked_box[0]), int(tracked_box[1]))
                p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Write the frame to output
            out.write(frame)

        cap.release()
        out.release()

    return jsonify({"message": "Video processed successfully", "video_url": f"/static/output_video_tracking.mp4"})

if __name__ == "__main__":
    app.run(debug=True)
