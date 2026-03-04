#!/usr/bin/env python3
"""
Kage Bunshin no Jutsu - Video Motion Detection Application

When you perform the Kage Bunshin hand sign, the video clones into 5 shadow clones!
Hand sign: Left hand middle finger + Right hand index finger forming a cross.
Alternative: Single hand with index and middle fingers extended (V/cross).
"""

import cv2
import numpy as np
from gesture_detector import KageBunshinGestureDetector
from person_segmenter import PersonSegmenter
import mediapipe as mp


def create_person_clone_layout(
    frame: np.ndarray, segmenter: PersonSegmenter, num_clones: int = 4
) -> np.ndarray:
    """
    Clone only the person (not the background) within the same frame.
    Uses segmentation to extract the person and places 5 copies in one frame.
    """
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask = segmenter.get_mask(frame_rgb)
    mask_3ch = np.stack([mask, mask, mask], axis=-1)

    # Person pixels only (background = black)
    person_bgr = (frame * mask_3ch).astype(np.uint8)
    alpha = (np.clip(mask, 0, 1) * 255).astype(np.uint8)

    # Start with full frame (original person in place)
    output = frame.copy()

    # Offsets for 4 clones: 2 on left, 2 on right (parallel, no overlap)
    # Horizontal: 200px separation from center; Vertical: 130px between top/bottom
    offsets = [
        (-200, -130),   # left-top
        (-200, 130),    # left-bottom
        (200, -130),    # right-top
        (200, 130),     # right-bottom
    ]

    for dx, dy in offsets:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        person_shifted = cv2.warpAffine(person_bgr, M, (w, h))
        alpha_shifted = cv2.warpAffine(alpha, M, (w, h))

        # Shadow clone effect: slight blue tint on clones
        hsv = cv2.cvtColor(person_shifted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) * 0.85, 0, 255).astype(np.uint8)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(np.int16) + 8, 0, 180).astype(np.uint8)
        person_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Alpha blend overlay
        alpha_3ch = np.stack([alpha_shifted / 255.0] * 3, axis=-1)
        valid = alpha_shifted > 15
        output = np.where(
            valid[:, :, np.newaxis],
            (output * (1 - alpha_3ch) + person_shifted * alpha_3ch).astype(np.uint8),
            output,
        ).astype(np.uint8)

    return output


def draw_hand_landmarks(frame, results):
    """Draw hand landmarks on frame for visual feedback."""
    if not results.multi_hand_landmarks:
        return frame
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )
    return frame


def main():
    print("=" * 50)
    print("  Kage Bunshin no Jutsu - Shadow Clone Detection")
    print("=" * 50)
    print("\nHand sign: Index+middle extended, ring+pinky folded, hands form cross!")
    print("\nHold the gesture to summon 5 shadow clones!")
    print("Press 'c' to manually trigger clones (for testing)")
    print("Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = KageBunshinGestureDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    segmenter = PersonSegmenter(model_selection=1)  # 1 = landscape, faster

    show_clones = False
    clone_duration = 0
    max_clone_duration = 90  # ~3 seconds at 30fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural feel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect gesture (or manual trigger with 'c')
            jutsu_detected = detector.detect(frame_rgb)

            if jutsu_detected:
                show_clones = True
                clone_duration = 0
                print("Kage Bunshin no Jutsu! Shadow clones summoned!")

            if show_clones:
                clone_duration += 1
                display_frame = create_person_clone_layout(frame, segmenter, num_clones=4)
                # Add jutsu text
                cv2.putText(
                    display_frame,
                    "KAGE BUNSHIN NO JUTSU!",
                    (display_frame.shape[1] // 2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Shadow Clones: 4",
                    (display_frame.shape[1] // 2 - 100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 255),
                    2,
                )
                if clone_duration >= max_clone_duration:
                    show_clones = False
            else:
                display_frame = frame.copy()
                # Draw hand landmarks for feedback
                results = detector.get_hand_landmarks(frame_rgb)
                display_frame = draw_hand_landmarks(display_frame, results)
                # Instruction overlay
                cv2.putText(
                    display_frame,
                    "Perform Kage Bunshin hand sign to clone!",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Index+middle up, ring+pinky folded, cross hands",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

            cv2.imshow("Kage Bunshin no Jutsu", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                # Manual trigger for testing
                show_clones = True
                clone_duration = 0
                print("Manual trigger: Shadow clones summoned!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        segmenter.close()
        print("\nJutsu session ended. See you next time!")


if __name__ == "__main__":
    main()
