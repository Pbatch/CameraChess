import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LABEL_STUDIO_DIR = os.path.join(ROOT_DIR, 'label_studio')
UPLOAD_DIR = os.path.join(LABEL_STUDIO_DIR, 'data', 'media', 'upload')
