"""
Node template for creating custom nodes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode

import mediapipe as mp
import numpy as np
from . import hand


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.direction = ""
        self.wrist_x = 0.0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs["img"]
        # print(type(img), img.shape)
        img.flags.writeable = False
        results = self.hands.process(img)

        width = img.shape[1]
        height = img.shape[0]
        blank_img = np.zeros((height, width, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            # handedness = results.multi_handedness
            # print(len(handedness))
            # print(handedness)
            # world_landmarks = results.multi_hand_world_landmarks
            # print(world_landmarks)
            if len(results.multi_handedness) > 0:
                # hand_landmarks = results.multi_hand_landmarks
                # print(len(hand_landmarks))
                # print(hand_landmarks)
                right_hand_landmark_list = hand.get_hand(results, "Right")
                if right_hand_landmark_list:
                    # print(f"{type(right_hand_landmark_list)}")
                    # print(dir(right_hand_landmark_list))

                    rh_landmark_container = right_hand_landmark_list.landmark
                    # print(type(rh_landmark_container))
                    # print(dir(rh_landmark_container))

                    print(rh_landmark_container[hand.WRIST])

                else:
                    print("None")

            self._draw_hands(blank_img, results)
            # img.flags.writeable = True
            # def_style_landmarks = (
            #     self.mp_drawing_styles.get_default_hand_landmarks_style()
            # )
            # def_style_conn = self.mp_drawing_styles.get_default_hand_connections_style()
            # hand_conn = self.mp_hands.HAND_CONNECTIONS
            # for hand_landmarks in results.multi_hand_landmarks:
            #     self.mp_drawing.draw_landmarks(
            #         blank_img,
            #         hand_landmarks,
            #         hand_conn,
            #         def_style_landmarks,
            #         def_style_conn,
            #     )

        outputs = {"img": blank_img}
        return outputs

    def _draw_hands(self, img, results):
        # img.flags.writeable = True
        def_style_landmarks = self.mp_drawing_styles.get_default_hand_landmarks_style()
        def_style_conn = self.mp_drawing_styles.get_default_hand_connections_style()
        hand_conn = self.mp_hands.HAND_CONNECTIONS
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                hand_conn,
                def_style_landmarks,
                def_style_conn,
            )


# Technotes:
#   results = hands.process(img)
#
#   results object data structure details
#       .multi_handedness           == python list of ClassificationList object
#       .multi_hand_landmarks       == NormalizedLandmarkList
#       .multi_hand_world_landmarks == LandmarkList
#
#       ClassificationList:
#           .classification     == a list of hand classification
#           .classification[0]  == Classification object
#
#       Classification:
#           .index  == index in list (above)
#           .label  == "Left" or "Right"
#           .score  == hand confidence
#
#
