import asyncio
import base64
import io
import json
import pickle

import numpy as np
import requests
import websockets
from PIL import Image

from camera_chess.classifier import Classifier
from camera_chess.state import State
from camera_chess.visualizer import Visualizer

# http://localhost:7272/chesshub => local connection string
# https://ChessVisualiserApi20230221132052.azurewebsites.net/chesshub => remote connection string (wss instead of ws)

negotiation = requests.post('http://localhost:7272/chesshub/negotiate?negotiateVersion=0').json()
classifier = Classifier(model_path='models/480S.xml',
                        keypoints=np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32),
                        conf_thres=0.1)
visualizer = Visualizer()


def toSignalRMessage(data):
    return f'{json.dumps(data)}\u001e'


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = base64.b64encode(buffer.getvalue())
    return image_bytes


def bytes_to_image(image_bytes):
    image_string = image_bytes.replace('data:image/jpeg;base64,', '')
    image = Image.open(io.BytesIO(base64.b64decode(image_string)))
    return image


async def connectToChessHub(connectionId):
    uri = f"ws://localhost:7272/chesshub?id={connectionId}"
    async with websockets.connect(uri) as websocket:

        async def handshake():
            await websocket.send(toSignalRMessage({"protocol": "json", "version": 1}))
            handshake_response = await websocket.recv()
            print(f"handshake_response: {handshake_response}")

        async def listen():
            while _running:
                try:
                    response = await websocket.recv()
                except websockets.ConnectionClosed as e:
                    print(f'Connected closed: {e}')
                    continue
                await process_response(response)

        async def process_response(response):
            if "ProcessImage" in response:
                image = response
                await process_image(image)
            elif "UploadImage" in response:
                print(response)

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
            try:
                keypoints = np.array([image_data[s][12:].split(':') for s in ['H1', 'A1', 'A8', 'H8']],
                                     dtype=np.float32)
            except Exception as e:
                print(e)
                return str(e)
            keypoints[..., 0] *= image.width
            keypoints[..., 1] *= image.height
            classifier.keypoints = keypoints
            classifier.set_kd_tree()
            state = State(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            pred = classifier.run(np.array(image))

            image = visualizer.add_bboxes(image, pred, keypoints)
            image = visualizer.add_board(image, state)

            output = {'image': image_to_bytes(image).decode(),
                      'state': pickle.dumps(state).decode("ISO-8859-1")}

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

            await websocket.send(toSignalRMessage(send_fen_message))

        await handshake()

        _running = True

        listen_task = asyncio.create_task(listen())

        await listen_task


print(f"connectionId: {negotiation['connectionId']}")
asyncio.run(connectToChessHub(negotiation['connectionId']))
