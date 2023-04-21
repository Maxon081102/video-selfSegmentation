import os
import cv2
import torch

def generate_annotate(img_name, mask_generator, mask_annotator, detector):
    img_bgr = cv2.imread(f"data/X/{img_name}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(img_rgb)
    detections = detector.from_sam(result)
    annotated_img = mask_annotator.annotate(image_bgr, detections)
    return annotated_img
