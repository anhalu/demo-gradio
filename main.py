import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import json

model = YOLO('yolov8n.pt')


def detect_image(image):
    predictions = model([image])
    results = []
    for prediction in predictions:
        boxes = prediction.boxes.xyxy.to(torch.int32).tolist()
        classes = prediction.boxes.cls.to(torch.int32).tolist()
        probs = prediction.boxes.conf.cpu().tolist()
        name_dict = prediction.names
        for cls, prob, box in zip(classes, probs, boxes):
            class_name = name_dict[cls]
            _json = {
                'label': class_name,
                'prob': prob,
                'box': box
            }
            results.append(_json)

    return json.dumps(results)


def test():
    img = cv2.imread('test.jpeg')
    res = detect_image(img)
    # print(res)


if __name__ == '__main__':
    # test()
    gr.Interface(
        detect_image,
        inputs="image",
        outputs="text",
    ).launch()
