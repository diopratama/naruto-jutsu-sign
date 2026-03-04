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
        self._frames_required = 3
        self._cooldown_frames = 0
        self._cooldown_duration = 25
        self._last_debug = {
            "hands": 0,
            "left_shape": False,
            "right_shape": False,
            "cross_ok": False,
            "orientation_ok": False,
            "cooldown": 0,
        }

        # Thresholds for cross formation (normalized 0-1 coords)
        self._cross_distance_threshold = 0.2

    @staticmethod
    def _distance(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    @staticmethod
    def _joint_angle(a, b, c) -> float:
        """
        Angle ABC in degrees using 2D image-space points.
        """
        ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
        bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_theta = float(np.dot(ba, bc) / (norm_ba * norm_bc))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _fingers_extended(self, landmarks) -> bool:
        """
        At least one crossing finger (index or middle) is extended.
        This is intentionally tolerant because one finger is often partially occluded
        in real crossed-hand poses.
        """
        return len(self._extended_tip_indices(landmarks)) >= 1

    def _extended_tip_indices(self, landmarks) -> list:
        """Return list of extended tip indices among index/middle."""
        wrist = landmarks[HandLandmark.WRIST]
        index_tip = landmarks[HandLandmark.INDEX_TIP]
        index_pip = landmarks[HandLandmark.INDEX_PIP]
        index_mcp = landmarks[HandLandmark.INDEX_MCP]
        middle_tip = landmarks[HandLandmark.MIDDLE_TIP]
        middle_pip = landmarks[HandLandmark.MIDDLE_PIP]
        middle_mcp = landmarks[HandLandmark.MIDDLE_MCP]

        index_angle = self._joint_angle(index_tip, index_pip, index_mcp)
        middle_angle = self._joint_angle(middle_tip, middle_pip, middle_mcp)

        index_reach = self._distance(index_tip, wrist) - self._distance(index_mcp, wrist)
        middle_reach = self._distance(middle_tip, wrist) - self._distance(middle_mcp, wrist)

        # Looser thresholds to support real-world webcam noise + partial occlusion.
        index_extended = index_angle > 132 and index_reach > 0.018
        middle_extended = middle_angle > 128 and middle_reach > 0.016

        tips = []
        if index_extended:
            tips.append(HandLandmark.INDEX_TIP)
        if middle_extended:
            tips.append(HandLandmark.MIDDLE_TIP)
        return tips

    def _fingers_folded(self, landmarks) -> bool:
        """
        Orientation-invariant folded check:
        folded fingers tend to have smaller PIP angles and less reach.
        """
        wrist = landmarks[HandLandmark.WRIST]
        ring_tip = landmarks[HandLandmark.RING_TIP]
        ring_pip = landmarks[HandLandmark.RING_PIP]
        ring_mcp = landmarks[HandLandmark.RING_MCP]
        pinky_tip = landmarks[HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[HandLandmark.PINKY_PIP]
        pinky_mcp = landmarks[HandLandmark.PINKY_MCP]

        ring_angle = self._joint_angle(ring_tip, ring_pip, ring_mcp)
        pinky_angle = self._joint_angle(pinky_tip, pinky_pip, pinky_mcp)

        ring_reach = self._distance(ring_tip, wrist) - self._distance(ring_mcp, wrist)
        pinky_reach = self._distance(pinky_tip, wrist) - self._distance(pinky_mcp, wrist)

        ring_folded = ring_angle < 145 or ring_reach < 0.015
        pinky_folded = pinky_angle < 145 or pinky_reach < 0.015
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
        return dx / dy < 0.9  # Allow slight diagonal tilt

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
        return dy / dx < 0.9  # Allow slight diagonal tilt

    def _hand_has_jutsu_shape(self, landmarks) -> bool:
        """Index+middle extended, ring+pinky folded."""
        return self._fingers_extended(landmarks) and self._fingers_folded(landmarks)

    def _cross_formed(self, left_landmarks, right_landmarks) -> bool:
        """
        Left hand's extended fingers within distance of right hand's extended fingers.
        Forms the characteristic "+" cross.
        """
        left_tips = self._extended_tip_indices(left_landmarks)
        right_tips = self._extended_tip_indices(right_landmarks)

        # Fallback: use both tips when extension classification is uncertain.
        if not left_tips:
            left_tips = [HandLandmark.INDEX_TIP, HandLandmark.MIDDLE_TIP]
        if not right_tips:
            right_tips = [HandLandmark.INDEX_TIP, HandLandmark.MIDDLE_TIP]

        min_dist = float("inf")
        for lt in left_tips:
            for rt in right_tips:
                d = self._distance(left_landmarks[lt], right_landmarks[rt])
                if d < min_dist:
                    min_dist = d

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
            self._last_debug.update({"cooldown": self._cooldown_frames})
            return False

        results = self.hands.process(frame_rgb)

        detected_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        self._last_debug.update(
            {
                "hands": detected_hands,
                "left_shape": False,
                "right_shape": False,
                "cross_ok": False,
                "orientation_ok": False,
                "cooldown": 0,
            }
        )

        if not results.multi_hand_landmarks or detected_hands != 2:
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
        best_candidate = None
        for left_lm, right_lm in orderings:
            left_shape = self._hand_has_jutsu_shape(left_lm)
            right_shape = self._hand_has_jutsu_shape(right_lm)
            left_horizontal = self._hand_orientation_horizontal(left_lm)
            right_vertical = self._hand_orientation_vertical(right_lm)
            cross_ok = self._cross_formed(left_lm, right_lm)

            # Shape + cross + orientation.
            # Accept canonical pair; otherwise allow if at least one hand has expected axis.
            orientation_ok = (left_horizontal and right_vertical) or (
                self._hand_orientation_vertical(left_lm) and self._hand_orientation_horizontal(right_lm)
            ) or left_horizontal or right_vertical

            candidate = {
                "left_shape": left_shape,
                "right_shape": right_shape,
                "cross_ok": cross_ok,
                "orientation_ok": orientation_ok,
            }
            if best_candidate is None:
                best_candidate = candidate
            else:
                prev_score = int(best_candidate["left_shape"]) + int(best_candidate["right_shape"]) + int(best_candidate["cross_ok"]) + int(best_candidate["orientation_ok"])
                cand_score = int(left_shape) + int(right_shape) + int(cross_ok) + int(orientation_ok)
                if cand_score > prev_score:
                    best_candidate = candidate

            if left_shape and right_shape and cross_ok and orientation_ok:
                is_jutsu = True
                best_candidate = candidate
                break

        if best_candidate:
            self._last_debug.update(best_candidate)

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

    def get_debug_status(self) -> dict:
        """Return last frame detector status for on-screen troubleshooting."""
        return dict(self._last_debug)

    def close(self):
        self.hands.close()
