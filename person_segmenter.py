"""
Person segmentation using MediaPipe Selfie Segmentation.
Extracts the person from the frame for shadow clone effect.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple


class PersonSegmenter:
    """Segments person from background using MediaPipe."""

    def __init__(self, model_selection: int = 0):
        # model_selection=1 is landscape, faster; 0 is general
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def get_mask(self, frame_rgb) -> np.ndarray:
        """
        Get segmentation mask: 0-1 float, 1=person, 0=background.
        Mask may be smaller than input; caller should resize if needed.
        """
        results = self.segmentation.process(frame_rgb)
        mask = results.segmentation_mask
        if mask is None:
            return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32)
        # Mask might be different size - resize to match frame
        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(
                mask, (frame_rgb.shape[1], frame_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        return np.clip(mask, 0, 1).astype(np.float32)

    def extract_person(self, frame_bgr, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract person with alpha. Returns (person_rgba, mask_3ch).
        """
        mask_3ch = np.stack([mask] * 3, axis=-1)
        person_bgr = (frame_bgr * mask_3ch).astype(np.uint8)
        alpha = (mask * 255).astype(np.uint8)
        return person_bgr, alpha

    def close(self):
        self.segmentation.close()
