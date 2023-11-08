from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        videofile = request.files['videofile']
        video_path = "./videos/" + videofile.filename
        videofile.save(video_path)

        return render_template('index.html', video_path=video_path)

    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', 0, type=str)
    return Response(detect_faces_in_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists('videos'):
        os.makedirs('videos')
    app.run(port=3000, debug=True)
