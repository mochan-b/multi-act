# Calculate how long it takes for the backbone to execute a single iteration

import time
import argparse
import torch
import torch.nn as nn

from torchvision.models import resnet18

from detr.models.backbone import build_backbone


def run_backbones_loop(backbones, image):
    outs = []
    for sensor_id, backbone in enumerate(backbones):
        backbone.eval()
        with torch.no_grad():
            img = image[:, sensor_id]
            out = backbone(img)
            outs.append(out)
    return outs


# def run_backbones(backbones, image):
# outs = [backbone(image[:, sensor_id]) for sensor_id, backbone in enumerate(backbones)]
# return outs

def run_backbones(backbones, image):
    streams = [torch.cuda.Stream() for _ in backbones]
    outs = [None] * len(backbones)

    for i, (backbone, stream) in enumerate(zip(backbones, streams)):
        backbone.eval()
        with torch.cuda.stream(stream):
            outs[i] = backbone(image[:, i])

    # Wait for all streams to complete
    torch.cuda.synchronize()
    return outs

class Backbones(nn.Module):
    def __init__(self, args, num_backbones):
        super().__init__()
        self.backbones_list = [build_backbone(args) for _ in range(num_backbones)]
        self.backbones = nn.ModuleList(self.backbones_list)
    
    def forward(self, x):
        return [backbone(x_) for backbone, x_ in zip(self.backbones, x)]

def time_backbone():
    # Create an args namespace object
    args = argparse.Namespace(
        hidden_dim=512,
        position_embedding='sine',
        lr_backbone=1e-05,
        masks=False,
        backbone='resnet18',
        dilation=False,
    )

    num_backbones = 10
    batch_size = 8
    image = torch.randn(batch_size, num_backbones, 3, 480, 640)

    backbones = Backbones(args, num_backbones)
    # for _ in range(num_backbones):
        # backbone = build_backbone(args)
        # backbones.append(backbone)
    # backbones_ml = nn.DataParallel(nn.ModuleList(backbones)).cuda()
    # backbones_ml = nn.DataParallel(backbones).cuda()
    backbones_ml = backbones.cuda()

    # Set torch to eval mode
    # for backbone in backbones_ml:
        # backbone.eval()

    img_p = image.permute(1, 0, 2, 3, 4).cuda()

    # do warmup
    with torch.no_grad():
        warmup_iters = 5
        for _ in range(warmup_iters):
            # run_backbones(backbones_ml, image)
            backbones_ml(img_p)

        timing_iters = 20
        start = time.time()
        for _ in range(timing_iters):
            # run_backbones(backbones_ml, image)
            out = backbones_ml(img_p)
        end = time.time()
        print(f'Backbone time: {(end - start) / timing_iters}')

    print(len(out))
    print(out[0][0][0].size())


class MultiCameraResNet(nn.Module):
    def __init__(self, num_cameras=3):
        super().__init__()
        self.models = nn.ModuleList([resnet18(pretrained=True) for _ in range(num_cameras)])
    
    def forward(self, x):
        # x should be a list or tuple of inputs, one for each camera
        return [model(input_) for model, input_ in zip(self.models, x)]

def time_resnet18():
    # Assuming you have 3 cameras
    num_cameras = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the multi-camera model
    multi_camera_model = MultiCameraResNet(num_cameras)
    multi_camera_model = nn.DataParallel(multi_camera_model).to(device)

    # Simulating inputs from 3 cameras (replace with your actual input processing)
    batch_size = 4
    input_shape = (3, 224, 224)  # Assuming ResNet50 input size
    inputs = [torch.randn(batch_size, *input_shape) for _ in range(num_cameras)]

    # Move inputs to the device
    inputs = [input_.to(device) for input_ in inputs]

    # Process inputs in parallel
    start = time.time()
    with torch.no_grad():
        outputs = multi_camera_model(inputs)
    end = time.time()
    print(f'ResNet18 time: {(end - start) / batch_size}')


def check_correctness():
    num_backbones = 7
    batch_size = 1
    image = torch.randn(num_backbones, batch_size, 3, 480, 640)

    args = argparse.Namespace(
        hidden_dim=512,
        position_embedding='sine',
        lr_backbone=1e-05,
        masks=False,
        backbone='resnet18',
        dilation=False,
    )

    backbones = Backbones(args, num_backbones)
    backbones_ml = nn.DataParallel(backbones).cuda()
    out = backbones_ml(image)

    bbl = backbones.backbones_list
    for i in range(num_backbones):
        bbl[i].eval()
        with torch.no_grad():
            out_single = bbl[i](image[i])
            assert torch.allclose(out_single, out[i], atol=1e-5)

    print(out[0][0][0].size())

if __name__ == '__main__':
    time_backbone()
    # time_resnet18()
    # check_correctness()
