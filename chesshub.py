import asyncio
import base64
import io
import json
import pickle

import numpy as np
import requests
import websockets
from PIL import Image

from camera_chess.detector import Detector
from camera_chess.state import State
from camera_chess.tracker.tracker import Tracker
from camera_chess.visualizer import Visualizer

# http://localhost:7272/chesshub => local connection string
# https://ChessVisualiserApi20230221132052.azurewebsites.net/chesshub => remote connection string (wss instead of ws)

detector = Detector(model_path='models/480S-quant.xml',
                    weights_path='models/480S-quant.bin',
                    keypoints=np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32))

visualizer = Visualizer()

while True:
    try:
        negotiation = requests.post('https://ChessVisualiserApi20230221132052.azurewebsites.net/chesshub/negotiate?negotiateVersion=0').json()
        #negotiation = requests.post('http://localhost:7272/chesshub/negotiate?negotiateVersion=0').json()
        
        def to_signalr_message(data):
            return f'{json.dumps(data)}\u001e'
        
        
        def image_to_bytes(image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = base64.b64encode(buffer.getvalue()).decode()
            return image_bytes
        
        
        def bytes_to_image(image_bytes):
            image_string = image_bytes.replace('data:image/jpeg;base64,', '')
            image = Image.open(io.BytesIO(base64.b64decode(image_string)))
            return image
        
        
        def obj_to_bytes(obj):
            return pickle.dumps(obj).decode("ISO-8859-1")
        
        
        def bytes_to_obj(bytes):
            return pickle.loads(bytes.encode("ISO-8859-1"))
        
        
        def load_state(latest_state):
            if len(latest_state):
                state = bytes_to_obj(latest_state)
            else:
                state = State(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            return state
        
        
        def load_tracker(latest_tracker, keypoints):
            if len(latest_tracker):
                tracker = bytes_to_obj(latest_tracker)
            else:
                tracker = Tracker(fps=1,
                                  keypoints=keypoints,
                                  track_low_thresh=0.1)
            return tracker
        
        
        async def connect_to_chess_hub(connection_id):
            uri = f"wss://ChessVisualiserApi20230221132052.azurewebsites.net/chesshub?id={connection_id}"
            #uri = f"ws://localhost:7272/chesshub?id={connection_id}"
            async with websockets.connect(uri) as websocket:
        
                async def handshake():
                    await websocket.send(to_signalr_message({"protocol": "json", "version": 1}))
                    handshake_response = await websocket.recv()
                    print(f"handshake_response: {handshake_response}")
        
                async def listen():
                    _running = True
                    while _running:
                        try:
                            response = await websocket.recv()
                            await process_response(response)
                        except Exception as e:
                            print(e)
                            _running = False
        
                async def process_response(response):
                    if "ProcessImage" in response:
                        image = response
                        await process_image(image)
                    elif "UploadImage" in response:
                        print(response)
        
                async def process_image(d):
                    data = json.loads(d[:-1])
                    image_data = json.loads(data['arguments'][0])
        
                    image = bytes_to_image(image_data['Image'])
                    keypoints = np.array([image_data[s][12:].split(':') for s in ['H1', 'A1', 'A8', 'H8']],
                                         dtype=np.float32)
                    keypoints[..., 0] *= image.width
                    keypoints[..., 1] *= image.height
                    state = load_state(image_data['LatestState'])
                    tracker = load_tracker(image_data['LatestTracker'], keypoints)
        
                    detector.keypoints = keypoints
                    detections = detector.run(np.array(image))
                    tracks = tracker.update(detections)
                    state.update(tracks)
                    if state.change:
                        print(state.last_move)
        
                    image = visualizer.add_bboxes_from_tracks(image, tracks, keypoints)
                    # image = visualizer.add_board(image, state)
        
                    output = {'image': image_to_bytes(image),
                              'state': obj_to_bytes(state),
                              'tracker': obj_to_bytes(tracker)}
        
                    await send_image_processed(json.dumps(output))
        
                async def send_image_processed(fen):
                    send_fen_message = {
                        "type": 1,
                        "invocationId": "invocation_id",
                        "target": "ImageProcessed",
                        "arguments": [
                            fen
                        ]
                    }
        
                    await websocket.send(to_signalr_message(send_fen_message))
        
                await handshake()
        
                listen_task = asyncio.create_task(listen())
        
                await listen_task

        print(f"connectionId: {negotiation['connectionId']}")
        asyncio.run(connect_to_chess_hub(negotiation['connectionId']))
                
    except Exception as e:
        print(e)


