from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
import time


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    R=random.randint(0, 255)
    G=random.randint(0, 255)
    B=random.randint(0, 255)



    CLASES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    colores = [(R,G,B) for i in CLASES ]

    net =   cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (800,600), fx=0.5, fy=0.5)
            (h,w)=img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 0.007843, (300,300), 127.5)
            net.setInput(blob)
            detections=net.forward()
            for i in np.arange(0, detections.shape[2]):
                confianza=detections[0,0,i,2]
                if confianza>0.5:
                    id = detections[0,0,i,1]
                    caja = detections[0,0,i,3:7] * np.array([w,h,w,h])
                    (startX, startY, endX, endY)= caja.astype("int")
                    cv2.rectangle(img, (startX-1, startY-40), (endX+1, startY-3), colores[int(id)], -1)
                    cv2.rectangle(img, (startX, startY), (endX, endY), colores[int(id)], 4)
                    cv2.putText(img, CLASES[int(id)], (startX+10, startY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))  

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break


@app.route('/video_feed')
def video_feed():
    return Response(gen(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port = 3000, debug = True)
