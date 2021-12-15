import cv2
import torch
import time
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from flask import Flask, render_template, Response, jsonify, request
from torchvision import transforms

from video_utils import show_text, get_byte_image, load_lip_detector, bodyFrames2LipFrames
from camera import VideoStream
from model_utils import load_model
from dataset.corpus import SSLCorpus

app = Flask(__name__)

# camera and recorder variables
video_camera = None
global_frame = None
status = 'loading'
timer = 0

# dataset variables
corpus = SSLCorpus('../CSLR/data/')
# data_frame = corpus.load_data_frame("test")
vocab = corpus.create_vocab("test")
transformer = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.CenterCrop([224,224]),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    global status
    global timer

    if video_camera == None:
        video_camera = VideoStream(src=0)

    json = request.get_json()
    record = json['record']

    if record == 'true':
        timer = 3
        status = 'starting'
        return jsonify(result='started')
    else:
        status = 'stopping'
        return jsonify(result='stopped')

def video_stream():
    global video_camera 
    global global_frame
    global status
    global timer
    global vocab

    if video_camera == None:
        video_camera = VideoStream(src=0).start()

    pnet, rnet, onet = load_lip_detector()
    model = load_model()
    model.to("cuda")
    model.eval()

    status = 'waiting'

    while True:
        if status == 'waiting':
            while True:
                # grab the frame from the threaded video file stream
                frame = video_camera.read()

                # add text, show the (mirrored) frame
                frame = show_text(cv2.flip(frame, 1), "green", "Waiting to start")
                frame = get_byte_image(frame)
                global_frame = frame
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
                # stop when starting to record
                if status == 'starting':
                    break

        if status == 'starting':
            target_time = time.time() + timer
            while True:
                # grab the frame from the threaded video file stream
                frame = video_camera.read()

                countdown = target_time - time.time()
                text = "Recording starts in " + str(int(countdown)+1) + " sec"

                # add text, show the (mirrored) frame
                frame = show_text(cv2.flip(frame, 1), "orange", text)
                frame = get_byte_image(frame)
                global_frame = frame
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
                # stop after countdown
                if timer > 0 and countdown <= 0.0:
                    status = 'recording'
                    break

        if status == 'recording':
            start = time.time()
            video_camera.start_recording()
            # loop over frames from the video file stream
            while True:
                # grab the frame from the threaded video file stream
                frame = video_camera.read()
                frame = cv2.flip(frame, 1)

                time_elapsed = time.time() - start
                text = "Recording " + str(int(time_elapsed)+1) + " sec"

                # add text, show the frame
                frame = show_text(frame, "red", text)
                frame = get_byte_image(frame)
                global_frame = frame
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

                # stop after nTimeDuration sec
                if status == 'stopping':
                    captured_frames = np.array(video_camera.stop())
                    print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
                        (time_elapsed, str(captured_frames.shape), captured_frames.shape[0]/time_elapsed))
                    status = 'translating'
                    break

        if status == 'translating':
            start = time.time()
            lip_frames = bodyFrames2LipFrames(captured_frames, pnet, rnet, onet)

            if type(lip_frames) != type(None):
                # preprocess the images
                frames = list(transformer(captured_frames))
                # get prediction from model
                with torch.no_grad():
                    prob = [lpi.exp().cpu().numpy() for lpi in model(frames)]    
                hyp = model.decode(prob, 10, 0.01, False, 8)
                hyp = [" ".join([vocab[i] for i in hi]) for hi in hyp]
                print(hyp)

                elapsed = time.time() - start
                print("\nTime taken for prediction: %.1f sec" % elapsed)
            status = 'waiting'

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)