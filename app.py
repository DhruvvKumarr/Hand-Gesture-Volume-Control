from flask import Flask, render_template, Response, jsonify, request
import cv2
from gesture_engine import GestureEngine

app = Flask(__name__)
engine = GestureEngine()
camera_on = False
latest_stats = {}

def generate_frames():
    global latest_stats
    while True:
        if camera_on:
            frame, stats = engine.get_frame()
            latest_stats = stats
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_camera")
def start_camera():
    global camera_on
    camera_on = True
    return "started"

@app.route("/stop_camera")
def stop_camera():
    global camera_on
    camera_on = False
    return "stopped"

@app.route("/stats")
def stats():
    return jsonify(latest_stats)

@app.route("/mapping_graph")
def mapping_graph():
    graph = engine.generate_mapping_graph()
    return {"image": graph}

@app.route("/variation_graph")
def variation_graph():
    graph = engine.generate_variation_graph()
    return {"image": graph}

@app.route("/params", methods=["POST"])
def params():
    data = request.json
    engine.update_params(
        float(data["detect"]),
        float(data["track"]),
        int(data["maxhands"])
    )
    return "updated"

if __name__ == "__main__":
    app.run(debug=True)
