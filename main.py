from flask import Flask, render_template, Response, send_file, jsonify, make_response
import cv2
import os
import io
from base64 import b64encode
from model_config import get_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img

app = Flask(__name__)

Class = ["with_mask", "without_mask"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
camera.set(3, 1920)
camera.set(4, 1080)
def gen_frame(begin):
    if begin == 1:
        global camera
        camera = cv2.VideoCapture(0)
        camera.set(3, 1920)
        camera.set(4, 1080)
        while begin == 1:
            global frame
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, img = cv2.imencode('.jpg', frame)
                img = img.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    else:
        camera.release()

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def index():
    return render_template('home.html', begin=0, capture_btn = False, captured_img = None)

@app.route("/start_video", methods=["POST","GET"])
def start_video():
    return render_template('home.html', begin = 1, capture_btn = True, captured_img = None)

@app.route('/video_feed/<int:begin>', methods=["POST","GET"])
def video_feed(begin):
    return Response(gen_frame(begin), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_video', methods=["POST","GET"])
def end_video():
    return render_template('home.html', begin= 0, capture_btn = False, captured_img = None)

app.config["CACHE_TYPE"] = "null"
@app.route('/capture', methods=["POST", "GET"])
def capture():
    global frame
    #print(frame, frame.shape)
    cv2.imwrite('./static/captured.jpg', frame)
    image_bn = open('./static/captured.jpg', 'rb').read()
    image = b64encode(image_bn).decode('utf-8')
    return jsonify({'status':1, 'image': image})

@app.route('/prediction', methods=["POST", "GET"])
def predict():
    
    model = get_model()
    img = cv2.imread('./static/captured.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = np.cast(gray, np.uint8)
    image = preprocess_input(img)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            print("here")
            face_img = image[x:x+w,y:y+h]
            print(face_img.shape)
            face_img = cv2.resize(face_img, (224, 224), interpolation = cv2.INTER_NEAREST)
            pred = model.predict(face_img[np.newaxis,:,:,:])
            print(str(pred[0,np.argmax(pred)]))
            accuracy = str(pred[0,np.argmax(pred)])
            cv2.putText(img, Class[np.argmax(pred)]+":"+ accuracy, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
    else:
        image = cv2.resize(image, (224, 224))
        pred = model.predict(image[np.newaxis,:,:,:])
        print(str(pred[0,np.argmax(pred)]))
        accuracy = str(pred[0,np.argmax(pred)])
        cv2.putText(img, Class[np.argmax(pred)]+":"+ accuracy, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite('./static/pred_img.jpg', img)
    pred_img = open('./static/pred_img.jpg', 'rb').read()
    image = b64encode(pred_img).decode('utf-8')
    return jsonify({'retry': 1, 'image': image})

@app.route('/delete_img', methods=["POST", "GET"])
def delete_img():
    os.remove("./static/captured.jpg")
    return render_template('home.html', begin = 1, capture_btn = True, captured_img = None)

@app.route('/retry', methods=['POST', 'GET'])
def retry():
    os.remove("./static/captured.jpg")
    os.remove("./static/pred_img.jpg")
    return render_template('home.html', begin = 1, capture_btn = True, captured_img = None)

app.run(debug=True)