label:
	docker run -it \
	-p 8080:8080 \
	-v C:\Users\Peter\PycharmProjects\CameraChess\label_studio\data:/label-studio/data \
	--env LABEL_STUDIO_USERNAME=peter_batchelor1@hotmail.com \
	--env LABEL_STUDIO_PASSWORD=password \
	--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
	-v C:\Users\Peter\PycharmProjects\CameraChess\label_studio\files:/label-studio/files \
	heartexlabs/label-studio:latest label-studio