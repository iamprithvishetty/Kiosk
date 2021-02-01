import RPi.GPIO as GPIO
import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from threading import Thread
from smbus2 import SMBus
from mlx90614 import MLX90614
import pigpio
from flask import Flask,redirect,url_for,render_template,make_response,Response
import json

app = Flask(__name__)
flask_value = ""

mask_count = 0
no_mask_count = 0
object_name=""

bus = SMBus(1)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

hand_check = 17
GPIO.setup(hand_check, GPIO.IN, GPIO.PUD_DOWN)

servo = 23
pwm = pigpio.pi() 
pwm.set_mode(servo, pigpio.OUTPUT)
pwm.set_PWM_frequency( servo, 50 )

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/data', methods=["GET", "POST"])
def data():
    global flask_value
    data = str(flask_value)
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

@app.route('/video_feed')
def video_feed():
    return Response(Mask_Check(), mimetype='multipart/x-mixed-replace; boundary=frame')

def Temp_Check():
    
    bus = SMBus(1)
    thermometer_address = 0x5a
    thermometer = MLX90614(bus,thermometer_address)
    #print(thermometer.get_ambient())
    print(thermometer.get_object_1())
    Temp_value = thermometer.get_object_1()
    bus.close()
    return Temp_value

def angle_to_percent(angle) :
    if angle > 180 or angle < 0 :
        return False

    start = 4.5
    end = 12
    ratio = (end - start)/180 #Calcul ratio from angle to percent

    angle_as_percent = angle * ratio

    return start + angle_as_percent

def Mask_Check():
    global object_name,mask_count,no_mask_count
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    '''parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)'''
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.8)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    '''parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')'''

    args = parser.parse_args()

    MODEL_NAME = "TFLiteModel"
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = False

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = "/home/pi/rpi-tensorflow-mask-detector/TFLiteModel/masktracker.tflite"

    # Path to label map file
    PATH_TO_LABELS = "/home/pi/rpi-tensorflow-mask-detector/TFLiteModel/masklabelmap.txt"

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                #print(object_name)
                if object_name=="mask":
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    mask_count=mask_count+1
                elif object_name == "no mask":
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 0, 255), 2)
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    no_mask_count=no_mask_count+1
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
 
def Sanitizer():
    while True:
        flag = 0 
        if GPIO.input(hand_check) == True:
            time_now= time.time()
            while GPIO.input(hand_check) == True:
                if time.time() - time_now > 3:
                    flag=1
                    break
                continue
            if flag == 1:
                flag=0
                print( "0 deg" )
                pwm.set_servo_pulsewidth( servo, 500 ) 
                time.sleep(1)
                 
                print( "180 deg" )
                pwm.set_servo_pulsewidth( servo, 2500 ) 
                time.sleep(1)
                
                print( "0 deg" )
                pwm.set_servo_pulsewidth( servo, 500 ) 
                time.sleep(1)
    
def Mask_Main():
    global flask_value,mask_count,no_mask_count,object_name
    while True:
        flask_value = "Welcome to Foster, saving lives"
        if mask_count>2 or no_mask_count>2:
            if object_name == "no mask":
                print("Please wear your mask")
                flask_value = "Please wear your mask"
            elif object_name == "mask":
                print("Thank you for wearing Mask")
                flask_value = "Thank you for wearing Mask"
            time.sleep(5)
            time_start = time.time()
            time_up = 0
            print("Please wait and stay still")
            flask_value="Please wait and stay still"
            while(mask_count<5 and time_up <20):
                time_up =time.time()-time_start
                
            if time_up > 20:
                print("Mask not detected")
                flask_value = "Mask not detected"
                flag_mask = 0
            elif mask_count>=5:
                print("Mask detected")
                flask_value = "Mask detected"
                flag_mask = 1
            time.sleep(3)
            if flag_mask == 1:
                print("Kindly Check your temperature")
                flask_value = "Kindly Check your temperature"
                Temperature = Temp_Check()
                time_start = time.time()
                time_up = 0
                while Temperature <31 and time_up < 10:
                    Temperature = Temp_Check()
                    time_up = time.time()-time_start
                    continue
                temp_list = []
                for max_temp in range(0,25):
                    time.sleep(0.15)
                    Temp_New = Temp_Check()
                    if Temp_New> Temperature:
                        Temperature= Temp_New
                    flask_value = str(Temp_New)[0:4]
                time.sleep(1)              
                if Temperature <34.0:
                    flag_temp = 1
                    print("Your temperature is normal")
                    flask_value = "Your temperature is normal"
                    time.sleep(3)
                else:
                    flag_temp = 0
                    print("Your temperature is high")
                    flask_value = "Your temperature is high"
                    time.sleep(3)
                    
                print("Kindly Santize your hand")
                flask_value = "Kindly Santize your hand"
                time.sleep(6)
                    
            mask_count=0
            no_mask_count = 0
            print("Thank You")
            flask_value = "Thank You"
            time.sleep(3)

Thread_Mask = Thread(target = Mask_Check, args = ())
Thread_Main_Mask = Thread(target = Mask_Main, args = ())
Thread_Sanitizer =  Thread(target= Sanitizer, args = ())
            
if __name__ == "__main__":
    Thread_Mask.start()
    Thread_Main_Mask.start()
    Thread_Sanitizer.start()
    app.run(debug=True)
    

