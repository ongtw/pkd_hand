"""
Node template for creating custom nodes.
"""

from typing import Any, Dict, List, Tuple
import itertools

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from hand import HANDS_DB, draw_hand, connect_db, read_hands, draw_text

import cv2
import numpy as np


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.conn = connect_db()
        self.df = read_hands(self.conn)
        self.num_hands = len(self.df)
        # self.test_df()
        self.idx_hand = 0
        self.auto_view = False

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs["img"]

        width = img.shape[1]
        height = img.shape[0]
        blank_img = np.zeros((height, width, 3), dtype=np.uint8)

        key = cv2.waitKey(20)  # poll key press

        # auto infinite loop
        idx_new = self.idx_hand + 1 if self.auto_view else None

        if key == ord("j"):
            idx_new = self.idx_hand - 1  # prev hand
        elif key == ord("k"):
            idx_new = self.idx_hand + 1  # next hand
        elif key == ord("h"):
            idx_new = self.idx_hand - 10  # prev hand by 10
        elif key == ord("l"):
            idx_new = self.idx_hand + 10  # next hand by 10
        elif key == ord("z"):
            self.auto_view = not self.auto_view

        if idx_new is not None:
            self.idx_hand = idx_new % self.num_hands

        norm_landmarks, gesture_id = self.get_landmarks(self.idx_hand)
        hand_landmarks_2d = self.convert_norm_landmarks_to_2d_coords(
            norm_landmarks, num_pixels=500, base_coord=(int(width / 2), 600)
        )
        # print(hand_landmarks_2d)

        draw_text(blank_img, 10, 30, f"{HANDS_DB} Viewer")

        draw_hand(blank_img, hand_landmarks_2d)

        info_text = (
            f"{self.idx_hand + 1} of {self.num_hands}, gesture_id={gesture_id}"
            f" {'[auto]' if self.auto_view else ''}"
        )
        draw_text(blank_img, 200, height - 10, info_text)

        outputs = {"img": blank_img}
        return outputs

    def get_landmarks(self, i: int) -> List[Any]:
        assert i >= 0
        df = self.df.iloc[[i]]
        gesture_id = df["gesture_id"].values[0]
        values = df.drop(["id", "gesture_id"], axis=1).values.flatten().tolist()
        landmarks = list(zip(values[::2], values[1::2]))
        return landmarks, gesture_id

    def test_df(self) -> None:
        print(self.df.info())
        df = self.df.iloc[[1]]
        print(f"gesture_id={df['gesture_id'].values[0]}")
        print("-----")
        norm_landmarks, gesture_id = self.get_landmarks(8)
        # print(norm_landmarks)
        hand_landmarks_2d = self.convert_norm_landmarks_to_2d_coords(
            norm_landmarks, num_pixels=500, base_coord=(880, 600)
        )
        print(hand_landmarks_2d)

    def convert_norm_landmarks_to_2d_coords(
        self, norm_landmarks, num_pixels: int, base_coord: Tuple[int, int]
    ) -> List[Any]:
        """Convert normalised hand landmarks into 2D (x,y) coordinates.

        Args:
            norm_landmarks (_type_): Normalised hand landmarks.
            num_pixels (int): Number of pixels represented by 1.0
            base_coord: (x,y) coords of wrist

        Returns:
            List[Any]: List of 2D (x,y) hand coordinates

        Technotes:
            - Range of normalised coords = [-1, 1]
            - (0,0) = Wrist of hand
            - Map normalised coords to pixel coords based on num_pixels,
              then convert those pixel coords (both +ve and -ve) to be
              absolute pixel coords anchored around wrist location given
              by base_coord
        """
        # convert to 1D list
        norm_landmarks_list = list(itertools.chain.from_iterable(norm_landmarks))
        # print(norm_landmarks_list)

        # calculate relative pixel coords
        rel_landmarks = [int(round(num_pixels * x, 0)) for x in norm_landmarks_list]

        # calculate actual pixel coords anchored around wrist
        base_x, base_y = base_coord
        # print(f"base_x={base_x}, base_y={base_y}")
        abs_x = map(lambda x: base_x + x, rel_landmarks[::2])
        abs_y = map(lambda y: base_y + y, rel_landmarks[1::2])
        landmark_points = list(map(list, zip(abs_x, abs_y)))
        # print(landmark_points)

        return landmark_points
