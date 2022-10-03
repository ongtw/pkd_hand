"""
Custom node to capture hand gestures into DB.
"""

from typing import Any, Dict
from functools import partial

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from hand import (
    connect_db,
    read_hand_catalog,
    save_hand,
    draw_text,
    calc_bounding_rect,
    draw_hand,
    project_hands_on_image,
)

import cv2
import mediapipe as mp


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.conn = connect_db()
        self._draw_msg_countdown = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs["img"]
        # print(type(img), img.shape)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # more accurate after BGR2RGB!
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)

        width = img.shape[1]
        height = img.shape[0]

        # poll key press
        key = cv2.waitKey(20)
        if key == ord("#"):
            res = read_hand_catalog(self.conn)
            self._draw_msg(10, 30, f"DB: {res}", countdown=60)

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
                draw_text(img, x1, y2, hand_label)

                if key > 0:
                    # NB: 0-9 + a-t (20 chars) for total of 30 labels
                    if key >= ord("0") and key <= ord("9"):
                        label = key - ord("0")
                    elif key >= ord("a") and key <= ord("t"):
                        label = key - ord("a") + 10  # offset by 10 digits above
                    else:
                        label = None
                    print(f"key={key}, label={label}")

                    if label is not None:
                        save_hand(self.conn, hand_landmarks_2d, label)
                        self._draw_msg(200, height - 10, f"save {label}", countdown=15)

        self._do_draw_msg(img)
        outputs = {"img": img}
        return outputs

    def _draw_msg(
        self,
        x: int,
        y: int,
        text: str,
        bgr: tuple = (255, 255, 255),
        size: float = 1.0,
        thickness: int = 2,
        countdown: int = 10,
    ):
        self._draw_msg_countdown = countdown
        self._draw_msg_func = partial(
            draw_text,
            x=x,
            y=y,
            text=text,
            bgr=bgr,
            size=size,
            thickness=thickness,
        )

    def _do_draw_msg(self, img):
        if self._draw_msg_countdown > 0:
            self._draw_msg_func(img)
            self._draw_msg_countdown -= 1
