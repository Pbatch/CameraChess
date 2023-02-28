import asyncio
import base64
import io
import json

import numpy as np
import requests
import websockets
from PIL import Image

from camera_chess.classifier import Classifier
from camera_chess.visualizer import Visualizer

# http://localhost:7272/chesshub => local connection string
# https://ChessVisualiserApi20230221132052.azurewebsites.net/chesshub => remote connection string (wss instead of ws)

negotiation = requests.post('http://localhost:7272/chesshub/negotiate?negotiateVersion=0').json()
classifier = Classifier(model_path='models/480S.xml',
                        keypoints=np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32),
                        conf_thres=0.1)


def toSignalRMessage(data):
    return f'{json.dumps(data)}\u001e'


async def connectToChessHub(connectionId):
    uri = f"ws://localhost:7272/chesshub?id={connectionId}"
    async with websockets.connect(uri) as websocket:

        async def handshake():
            await websocket.send(toSignalRMessage({"protocol": "json", "version": 1}))
            handshake_response = await websocket.recv()
            print(f"handshake_response: {handshake_response}")

        async def listen():
            while _running:
                response = await websocket.recv()
                await process_response(response)

        async def process_response(response):
            if "ProcessImage" in response:
                image = response
                await process_image(image)
            elif "UploadImage" in response:
                print(response)

        async def process_image(image):
            data = json.loads(image[:-1])
            image_data = json.loads(data['arguments'][0])
            pil_image = Image.open(io.BytesIO(base64.b64decode(image_data['Image'])))

            keypoints = np.array([image_data[s][12:].split(':') for s in ['H1', 'A1', 'A8', 'H8']], dtype=np.float32)
            keypoints[..., 0] *= pil_image.width
            keypoints[..., 1] *= pil_image.height
            classifier.keypoints = keypoints
            classifier.set_kd_tree()
            pred = classifier.run(pil_image)

            visualizer = Visualizer(keypoints)
            pil_image = visualizer.add_bboxes(pil_image, pred)
            pil_image.show()
            await send_image_processed(str(pred))

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
