import logging
import os

# Set tensorflow logging to only fatal logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import time
import numpy as np

from flask import Flask, render_template, Response, jsonify, request
from turbo_flask import Turbo

from video_utils import get_byte_image
from camera import VideoStream
from model_utils import PyModel

# Start a turbo flask app
app = Flask(__name__)
turbo = Turbo(app)

# global variables
status = "stopping"
video_camera = None
prediction = "පරිවර්තනය මෙතනින් දිස්වේ"
# init_time = time.time() # Note: comment/remove after calculating avg model load time
model = PyModel()
# model_load_time = time.time() - init_time  # Note: comment/remove after calculating avg model load time
# print("Time taken for loading the model: %.1f sec" % model_load_time) # Note: comment/remove after calculating avg model load time
timer = counter = 5
status_text = "කරුණාකර මොහොතක් ඉන්න"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    global status

    if video_camera == None:
        video_camera = VideoStream(src=0)

    json = request.get_json()
    record = json['record']

    if record == 'true':
        status = 'starting'
        return jsonify(result='started')

def video_stream():
    global video_camera 
    global prediction
    global model
    global status
    global vocab
    global timer
    global status_text
    global counter

    if video_camera == None:
        video_camera = VideoStream(src=0)
    
    while True:
        video_camera.start()
        # change status text
        status_text = "පරිවර්තනය ආරම්භ කළ හැක"
        update_status_text()
        update_button()
        while True:
            # grab and show the frame from the threaded video file stream
            frame = video_camera.read()
            frame = cv2.flip(frame, 1)
            frame = get_byte_image(frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
            # stop when starting to record
            if status == 'starting':
                break

        target_time = time.time() + timer
        prediction = "පරිවර්තනය මෙතනින් දිස්වේ"
        update_prediction()

        while True:
            # change status text
            countdown = target_time - time.time()
            status_text = "පටිගත කිරීම තත්පර " + str(int(countdown)+1) + " කින් ආරම්භ වේ"
            update_status_text()
            counter = int(countdown) + 1
            update_counter()

            # grab and show the frame from the threaded video file stream
            frame = video_camera.read()
            frame = cv2.flip(frame, 1)
            frame = get_byte_image(frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
            # stop after countdown
            if countdown <= 0.0:
                break

        # start recorder and set appropriate instance variables
        start = time.time()
        time_elapsed = prev_time = 0
        video_camera.start_recording()
        status_text = "පටිගත වෙමින් පවතියි"
        update_status_text()
        update_to_empty_counter()
        current_frame = previous_frame = video_camera.read()

        # loop over frames from the video file stream
        while True:
            frame = cv2.flip(current_frame, 1)
            frame = get_byte_image(frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            # calculate frame difference each passing second after 2 sec of recording
            if int(time_elapsed) > 2 and int(time_elapsed) > int(prev_time):
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
                _, bw = cv2.threshold(frame_diff, 127, 255, cv2.THRESH_BINARY)
                # stop recording if frame_diff is very low
                if bw.mean() < 0.01:
                    status = "stopping"
                previous_frame = current_frame.copy()
                prev_time = time_elapsed

            # stop recording indefinitely after 10 sec
            if int(time_elapsed) == 8:
                status = "stopping"

            # stop the recording
            if status == 'stopping':
                captured_frames = np.array(video_camera.stop_recording())
                print(captured_frames.shape)
                # print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
                #     (time_elapsed, str(captured_frames.shape), captured_frames.shape[0]/time_elapsed)) # Note: comment/remove after calculating avg prediction times
                break

            current_frame = video_camera.read()
            time_elapsed = time.time() - start

        # change status text
        status_text = "පරිවර්තනය වෙමින් පවතියි... කරුණාකර මොහොතක් ඉන්න"
        update_status_text()

        # DEBUG CODE TO USE FILE FRAMES AS INPUT -------------------- 

        # liFiles = sorted(glob.glob("frames/*.jpg"))
        # liFrames = []
        # # loop through frames
        # for frame in liFiles:
        #     arFrame = cv2.imread(frame)
        #     liFrames.append(arFrame)

        # captured_frames = np.array(liFrames)

        # END OF DEBUG CODE -----------------------------------------

        # get prediction
        results = model.predict(captured_frames)

        if type(results['prediction']) != type(None):
            # Note: comment/remove after calculating avg prediction times -------------------------------------------

            # print("\nTime taken for lip frame extraction: %.1f sec" % results['lip_extraction_time'])
            # let_psv = results['lip_extraction_time']/time_elapsed
            # print("Time taken for lip frame extraction (per sec. of input video): %.1f sec" % let_psv)
            # print("\nTime taken for frame processing and prediction: %.1f sec" % results['prediction_time'])
            # pt_psv = results['prediction_time']/time_elapsed
            # print("Time taken for frame processing and prediction (per sec. of input video): %.1f sec" % pt_psv)

            # End Note ----------------------------------------------------------------------------------------------

            prediction = results['prediction']
        # elif results['prediction'] == '':
        #     prediction = "පරිවර්තනය අසාර්ථකයි.. නැවත උත්සාහ කරන්න"
        else:
            prediction = "පරිවර්තනයේදී දෝශයක් ඇතිවිය.. නැවත උත්සාහ කරන්න"
        update_prediction()

@app.context_processor
def inject_load():
    global prediction
    global status_text
    global counter
    return { 
        'prediction' : prediction,
        'status_text' : status_text,
        'counter' : counter
    }

def update_prediction():
    with app.app_context():
        turbo.push(turbo.replace(render_template('prediction.html'), 'prediction'))

def update_status_text():
    with app.app_context():
        turbo.push(turbo.replace(render_template('status.html'), 'status'))

def update_button():
    with app.app_context():
        turbo.push(turbo.replace(render_template('predict_button.html'), 'predict_button'))

def update_counter():
    with app.app_context():
        turbo.push(turbo.replace(render_template('counter.html'), 'counter'))

def update_to_empty_counter():
    with app.app_context():
        turbo.push(turbo.replace(render_template('empty_counter.html'), 'counter'))

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

# AVG MODEL LOAD TIME = 3.12s
# AVG LIP EXT TIME = 0.66s per sec of video
# AVG TIME FOR PREPROCESSING AND PREDICTION = 0.2s per sec of video