import json
import os

from PIL import Image


def main():
    export_id = 'project-1-at-2023-02-17-13-10-401bb257'
    dataset = 'hikaru_sarin'
    with open(os.path.join(f'label_studio/data/export/{export_id}.json')) as f:
        labels = json.load(f)

    for i, label in enumerate(labels[:20]):
        new_label = {'keypoints': {s: [] for s in ['h1', 'a1', 'a8', 'h8']},
                     'bboxes': []}
        for annotation in label['annotations'][0]['result']:
            value, width, height, type_ = [annotation[s] for s in ['value', 'original_width',
                                                                   'original_height', 'type']]
            if type_ == 'keypointlabels':
                key = value['keypointlabels'][0]
                x = float(value['x']) / 100
                y = float(value['y']) / 100
                new_label['keypoints'][key] = [x, y]
            elif type_ == 'rectanglelabels':
                class_ = value['rectanglelabels'][0]
                bbox = [class_] + [float(value[s]) / 100 for s in ['x', 'y', 'width', 'height']]
                new_label['bboxes'].append(bbox)

        new_label_path = f'data/{dataset}/labels/{i}.json'
        with open(new_label_path, 'w') as f:
            json.dump(new_label, f, indent=4)

        basename = os.path.basename(label['data']['img'])
        image = Image.open(os.path.join('label_studio', 'files', 'images', basename))
        image = image.convert('RGB')
        new_image_path = f'data/{dataset}/images2/{i}.jpg'
        image.save(new_image_path)


if __name__ == '__main__':
    main()
