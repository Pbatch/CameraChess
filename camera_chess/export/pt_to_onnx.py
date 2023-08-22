import torch

from camera_chess.export.wrapped_model import load_model


def export(model, save_path, dynamic=True, opset_version=16):
    image = torch.randint(0, 256, (8, 400, 500, 3), dtype=torch.uint8)
    if dynamic:
        dynamic_axes = {'images': {0: 'batch', 1: 'height', 2: 'width'},
                        'detections': {0: 'n_dets', 1: 'batch_bbox_conf_cls'}}
    else:
        dynamic_axes = None

    torch.onnx.export(model=model,
                      args=image.cuda(),
                      f=save_path,
                      verbose=False,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['detections'],
                      dynamic_axes=dynamic_axes)


def main():
    model_path = 'models/480L.pt'
    model = load_model(model_path)
    export(model, 'models/480L.onnx')


if __name__ == '__main__':
    main()
