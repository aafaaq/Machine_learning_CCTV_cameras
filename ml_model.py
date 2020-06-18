import cv2
import numpy
import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import PIL

print("loading model")
model = torch.load("detr_model.pth")


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def ml_frame(frame):
    img = Image.fromarray(frame).resize((800, 600))
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    img_tens = transform(img).unsqueeze(0).cpu()

    with torch.no_grad():
        output = model(img_tens)

    output['pred_boxes'][0].shape
    output['pred_logits'][0].argmax(-1)

    im2 = img.copy()
    for logits, boxes in zip(output['pred_logits'][0], output['pred_boxes'][0]):
        cls = logits.argmax()
        if cls >= len(CLASSES):
            continue
        label = CLASSES[cls]


    im2 = img.copy()
    drw = ImageDraw.Draw(im2)
    for logits, box in zip(output['pred_logits'][0], output['pred_boxes'][0]):
        cls = logits.argmax()
        if cls >= len(CLASSES):
            continue
        label = CLASSES[cls]
        box = box.cpu() * torch.Tensor([800, 600, 800, 600])
        x, y, w, h = box
        x0, x1 = x - w // 2, x + w // 2
        y0, y1 = y - h // 2, y + h // 2
        drw.rectangle([x0, y0, x1, y1], outline='red', width=3)
        drw.text((x - 10, y - 10), label, fill='white')
    model_frame = numpy.array(im2)
    return model_frame


