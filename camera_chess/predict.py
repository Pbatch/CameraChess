from ultralytics import YOLO

model = YOLO(model='data/best.pt')

image_path = 'data/object_detection/roboflow_5/images/20220602_205527_jpg.rf.ee7e22c7c5165377b2e3448ff2a94200.jpg'
model.predict(image_path, save=True)
