from flask import Flask,redirect,url_for,render_template,make_response,Response
from threading import Thread
import json
import time
import cv2

value="Hello World"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/data', methods=["GET", "POST"])
def data():
    global value
    data = str(value)
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

@app.route('/video_feed')
def video_feed():
    return Response(show_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def valuechange():
    global value
    for i in range(0,1000):
        value=i
        time.sleep(1)

def show_video():
    cap = cv2.VideoCapture(0)
    while True:
        success,frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

Thread_print = Thread(target=valuechange,args=[])

if __name__=="__main__":
    Thread_print.start()
    app.run(debug=True)