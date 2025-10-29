import numpy as np
import torch
import cv2
import base64
from threading import Thread
import time

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


from flask import Flask, request
from flask_socketio import SocketIO, send, emit, join_room, leave_room

use_sim3=False
submap_size=16
max_loops=1
min_disparity=50.0
conf_threshold=25.0


use_optical_flow_downsample = False
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting model loading")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)

print("Finish model loading\nStarting solver init")

solver = Solver(
    init_conf_threshold=conf_threshold,
    use_point_map=False,
    use_sim3=use_sim3,
    gradio_mode=False
)
print("Finish solver init\nStarting app init")

image_names_subset = []

keep_running_slam = True

def run_slam(): 
    global keep_running_slam
    while keep_running_slam:
        print("running slam thread function")

        global image_names_subset
        global solver

        try: 
            if len(image_names_subset) >= submap_size + 1:
                print("Running predictions")
                predictions = solver.run_predictions(image_names_subset[:submap_size+1], model, max_loops)
                solver.add_points(predictions)
                solver.graph.optimize()
                solver.map.update_submap_homographies(solver.graph)

                image_names_subset = image_names_subset[submap_size:]

                solver.update_all_submap_vis()
            else: 
                time.sleep(0.5)
        except:
            break
         


app = Flask(__name__)
app.config["secret_key"] = 'your_secret_key'

socketio = SocketIO(app)

frame_counter = 0 

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('last_frame')
def handle_run_slam():    
    print("last frame notified")

    global image_names_subset
    if len(image_names_subset) > 1: 
        print(image_names_subset)
        predictions = solver.run_predictions(image_names_subset, model, max_loops)
        solver.add_points(predictions)
        solver.graph.optimize()
        solver.map.update_submap_homographies(solver.graph)

        image_names_subset = image_names_subset[-1:]
    
    solver.update_all_submap_vis()


@socketio.on("disconnect")
def handle_disconnect():    
    print("disconnect")


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Expecting data as base64-encoded JPEG image string.
    """
    global frame_counter
    global image_names_subset

    try:
        # Decode base64 -> bytes
        frame_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Save frame
        image_name = f"recv_images/frame{frame_counter:04d}.jpg"
        cv2.imwrite(image_name, frame)
        frame_counter += 1

        print(f"Saved: {image_name}")

        print(image_name)
        if use_optical_flow_downsample:
            enough_disparity = solver.flow_tracker.compute_disparity(frame, min_disparity, False)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap
        

    except Exception as e:
        print(f"Error decoding frame: {e}")



if __name__ == '__main__':
    t = Thread(target=run_slam)
    t.start()
    print("Listening for video stream via Socket.IO...")
    socketio.run(app, host='0.0.0.0', port=9999)
