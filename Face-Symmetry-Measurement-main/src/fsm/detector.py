"""
detector.py -- thin wrapper around the MediaPipe Face Landmarker (Tasks API).

The model file models/face_landmarker.task is Google's published Face Landmarker
bundle (Apache-2.0). It is loaded once per process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

from .mesh import ROOT  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


@dataclass
class Detection:
    landmarks: np.ndarray      # (478, 2) pixel coordinates
    depth: np.ndarray          # (478,) relative depth (MediaPipe z, image-width units)
    transform: np.ndarray      # (4, 4) facial transformation matrix (canonical -> camera)
    n_faces: int               # number of faces found (we analyse the first)


@lru_cache(maxsize=1)
def _landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Download it from {MODEL_URL}")
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=2,                                  # 2 so that "multiple faces" can be reported
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,   # used for yaw / pitch / roll
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def detect(img_bgr: np.ndarray) -> Detection | None:
    """Run the landmarker on a BGR image. Returns None if no face is found."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    face = res.face_landmarks[0]
    xy = np.array([[p.x * w, p.y * h] for p in face], dtype=np.float64)
    z = np.array([p.z for p in face], dtype=np.float64)
    T = (np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
         if res.facial_transformation_matrixes else np.eye(4))
    return Detection(xy, z, T, len(res.face_landmarks))


def pose_from_transform(T: np.ndarray) -> dict:
    """
    Yaw / pitch / roll in degrees from the rotation part of the facial transformation
    matrix (rotation of the canonical face model into camera space, ZYX convention).
    Signs depend on the camera convention; the gate only uses magnitudes for yaw and
    pitch. Validated in tests by the flip test (yaw and roll change sign under flip).
    """
    R = T[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
