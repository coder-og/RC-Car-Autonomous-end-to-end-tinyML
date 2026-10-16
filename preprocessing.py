"""Shared OpenCV-order crop/resize for training, validation and inference."""
import cv2
import params

def preprocess_frame(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != params.inputchannels:
        raise ValueError("Expected a three-channel image")
    if not 0 <= params.crop_y1 < params.crop_y2 <= frame.shape[0]:
        raise ValueError("Crop bounds must lie inside the image")
    return cv2.resize(frame[params.crop_y1:params.crop_y2, :], params.inputres)
