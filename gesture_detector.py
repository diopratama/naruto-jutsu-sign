"""
Kage Bunshin (Shadow Clone) Jutsu Gesture Detector

Detects the canonical Kage Bunshin hand sign using landmark-based logic:
- Index (8) and Middle (12) fingers extended (tips higher than MCP 5, 9)
- Ring (16) and Pinky (20) fingers folded (tips below knuckles)
- Right hand: wrist-to-index vector mostly vertical
- Left hand: wrist-to-index vector mostly horizontal
- Cross: left and right extended fingers within distance threshold
"""

import mediapipe as mp
import numpy as np


# MediaPipe hand landmark indices (0-20)
class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_IP = 2
    THUMB_TIP = 3
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20


class KageBunshinGestureDetector:
    """Detects the Kage Bunshin hand sign using MediaPipe landmark logic."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.4):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._frames_detected = 0
        self._frames_required = 4
        self._cooldown_frames = 0
        self._cooldown_duration = 25

        # Thresholds for cross formation (normalized 0-1 coords)
        self._cross_distance_threshold = 0.2

    def _fingers_extended(self, landmarks) -> bool:
        """
        Index (8) and Middle (12) tips must be significantly higher than MCP (5, 9).
        Higher = smaller y in image coordinates.
        """
        index_tip = landmarks[HandLandmark.INDEX_TIP]
        index_mcp = landmarks[HandLandmark.INDEX_MCP]
        middle_tip = landmarks[HandLandmark.MIDDLE_TIP]
        middle_mcp = landmarks[HandLandmark.MIDDLE_MCP]

        index_raised = index_tip.y < index_mcp.y - 0.02  # tip higher than knuckle
        middle_raised = middle_tip.y < middle_mcp.y - 0.02
        return index_raised and middle_raised

    def _fingers_folded(self, landmarks) -> bool:
        """
        Ring (16) and Pinky (20) tips should be below their knuckles (PIP).
        Below = larger y (folded toward palm).
        """
        ring_tip = landmarks[HandLandmark.RING_TIP]
        ring_pip = landmarks[HandLandmark.RING_PIP]
        pinky_tip = landmarks[HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[HandLandmark.PINKY_PIP]

        ring_folded = ring_tip.y > ring_pip.y - 0.02
        pinky_folded = pinky_tip.y > pinky_pip.y - 0.02
        return ring_folded and pinky_folded

    def _hand_orientation_vertical(self, landmarks) -> bool:
        """
        Right hand: wrist-to-index vector mostly vertical.
        Vertical = |dx| small relative to |dy|.
        """
        wrist = landmarks[HandLandmark.WRIST]
        index_tip = landmarks[HandLandmark.INDEX_TIP]
        dx = abs(index_tip.x - wrist.x)
        dy = abs(index_tip.y - wrist.y)
        if dy < 0.02:
            return False
        return dx / dy < 0.6  # More vertical than horizontal

    def _hand_orientation_horizontal(self, landmarks) -> bool:
        """
        Left hand: wrist-to-index vector mostly horizontal.
        Horizontal = |dy| small relative to |dx|.
        """
        wrist = landmarks[HandLandmark.WRIST]
        index_tip = landmarks[HandLandmark.INDEX_TIP]
        dx = abs(index_tip.x - wrist.x)
        dy = abs(index_tip.y - wrist.y)
        if dx < 0.02:
            return False
        return dy / dx < 0.6  # More horizontal than vertical

    def _hand_has_jutsu_shape(self, landmarks) -> bool:
        """Index+middle extended, ring+pinky folded."""
        return self._fingers_extended(landmarks) and self._fingers_folded(landmarks)

    def _cross_formed(self, left_landmarks, right_landmarks) -> bool:
        """
        Left hand's extended fingers within distance of right hand's extended fingers.
        Forms the characteristic "+" cross.
        """
        left_index = left_landmarks[HandLandmark.INDEX_TIP]
        left_middle = left_landmarks[HandLandmark.MIDDLE_TIP]
        right_index = right_landmarks[HandLandmark.INDEX_TIP]
        right_middle = right_landmarks[HandLandmark.MIDDLE_TIP]

        # Check distance between left fingers and right fingers
        d_left_idx_right_idx = np.sqrt(
            (left_index.x - right_index.x) ** 2 + (left_index.y - right_index.y) ** 2
        )
        d_left_idx_right_mid = np.sqrt(
            (left_index.x - right_middle.x) ** 2 + (left_index.y - right_middle.y) ** 2
        )
        d_left_mid_right_idx = np.sqrt(
            (left_middle.x - right_index.x) ** 2 + (left_middle.y - right_index.y) ** 2
        )
        d_left_mid_right_mid = np.sqrt(
            (left_middle.x - right_middle.x) ** 2 + (left_middle.y - right_middle.y) ** 2
        )

        min_dist = min(
            d_left_idx_right_idx, d_left_idx_right_mid,
            d_left_mid_right_idx, d_left_mid_right_mid
        )
        return min_dist < self._cross_distance_threshold

    def _get_handedness(self, results) -> list:
        """Get list of 'Left' or 'Right' for each detected hand."""
        handedness_list = []
        if not results.multi_handedness:
            return handedness_list
        for h in results.multi_handedness:
            label = h.classification[0].label
            handedness_list.append(label)
        return handedness_list

    def detect(self, frame_rgb) -> bool:
        """
        Process frame and return True if Kage Bunshin jutsu is detected.
        """
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return False

        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            self._frames_detected = 0
            return False

        hands = results.multi_hand_landmarks
        handedness = self._get_handedness(results)

        # Match hands by handedness; try both orderings if handedness unclear
        left_landmarks = None
        right_landmarks = None
        for i, label in enumerate(handedness):
            if i < len(hands):
                if label == "Left":
                    left_landmarks = hands[i].landmark
                else:
                    right_landmarks = hands[i].landmark

        # Fallback: try both orderings (handedness can be unreliable)
        orderings = []
        if left_landmarks and right_landmarks:
            orderings = [(left_landmarks, right_landmarks)]
        else:
            orderings = [
                (hands[0].landmark, hands[1].landmark),
                (hands[1].landmark, hands[0].landmark),
            ]

        is_jutsu = False
        for left_lm, right_lm in orderings:
            left_shape = self._hand_has_jutsu_shape(left_lm)
            right_shape = self._hand_has_jutsu_shape(right_lm)
            left_horizontal = self._hand_orientation_horizontal(left_lm)
            right_vertical = self._hand_orientation_vertical(right_lm)
            cross_ok = self._cross_formed(left_lm, right_lm)

            # Shape + cross + orientation (prefer both; accept if at least one correct)
            orientation_ok = left_horizontal and right_vertical or left_horizontal or right_vertical
            if left_shape and right_shape and cross_ok and orientation_ok:
                is_jutsu = True
                break

        if is_jutsu:
            self._frames_detected += 1
            if self._frames_detected >= self._frames_required:
                self._frames_detected = 0
                self._cooldown_frames = self._cooldown_duration
                return True
        else:
            self._frames_detected = 0

        return False

    def get_hand_landmarks(self, frame_rgb):
        """Get raw hand landmarks for drawing."""
        return self.hands.process(frame_rgb)

    def close(self):
        self.hands.close()
