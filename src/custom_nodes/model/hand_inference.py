"""
Node template for creating custom nodes.
"""

from typing import Any, Dict, Tuple

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from hand import (
    draw_text,
    calc_bounding_rect,
    draw_hand,
    normalize_hand_landmarks,
    project_hands_on_image,
)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_SAVE_PATH = "hand_gesture_classifier.hdf5"


def Point(x: float, y: float) -> Tuple[int, int]:
    return (int(x), int(y))


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs["img"]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # more accurate after BGR2RGB!
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                hand_label = handedness.classification[0].label
                hand_landmarks_2d = project_hands_on_image(img, hand_landmarks)
                draw_hand(img, hand_landmarks_2d)

                # draw hand bbox
                [x1, y1, x2, y2] = calc_bounding_rect(img, hand_landmarks)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0))
                draw_text(img, x1, y2, f"{hand_label}")

                # inference hand
                norm_landmarks = normalize_hand_landmarks(hand_landmarks_2d)
                inf_input = np.array([norm_landmarks])
                predicts = self.model.predict(inf_input, verbose=0)
                predicts_flatten = np.squeeze(predicts)
                i = np.argmax(predicts_flatten)
                prob = predicts_flatten[i]

                if prob >= 0.95:
                    draw_text(img, x1, y1, f"Predict: {i}, {prob*100:.0f}%")

        outputs = {"img": img}
        return outputs
