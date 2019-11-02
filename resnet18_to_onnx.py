import torch
from torch.onnx import export
from torchvision.models import resnet18


if __name__ == '__main__':
    model = resnet18(pretrained=True)
    model = model.eval()
    sample_input = torch.randn(1, 3, 256, 256)
    export(
        model, 
        sample_input, 
        'resnet18.onnx', 
        verbose=True, 
        input_names=['model_input1'], 
        output_names=['output1']
    )