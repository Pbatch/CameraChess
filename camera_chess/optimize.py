from speedster import optimize_model, save_model
from ultralytics import YOLO
import torch


class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x, *args, **kwargs):
        res = self.model(x)
        return res[0], *tuple(res[1])


class OptimizedYOLO(torch.nn.Module):
    def __init__(self, optimized_model):
        super().__init__()
        self.model = optimized_model

    def forward(self, x, *args, **kwargs):
        res = self.model(x)
        return res[0], list(res[1:])


def main():
    yolo = YOLO('data/best.pt')
    model_wrapper = YOLOWrapper(yolo)

    # Provide some input data for the model
    input_data = [((torch.randn(1, 3, 640, 640),), torch.tensor([0])) for _ in range(100)]

    # Run Speedster optimization
    optimized_model = optimize_model(model_wrapper,
                                     input_data=input_data,
                                     metric_drop_ths=0.1,
                                     optimization_time='unconstrained',
                                     store_latencies=True,
                                     device="cpu")
    save_model(optimized_model, 'data/optimized')


if __name__ == '__main__':
    main()