"""
Node template for creating custom nodes.
"""

from typing import Any, Dict, List, Tuple
from functools import partial
import itertools

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from hand import (
    HANDS_DB,
    _BLUE,
    _WHITE,
    draw_hand,
    connect_db,
    read_hand_catalog,
    read_hands,
    draw_text,
)

import cv2
import numpy as np

# Constants
BTN_WIDTH = 100
BTN_HEIGHT = 50


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"mouse click: {x}, {y}, param={param}")
        self = param  # self is PKD's node object
        # print(type(self))
        # print(dir(self))
        for _, btn in self.buttons.items():
            if btn.is_clicked(x, y):
                btn.callback()
                return


class MyButton:
    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        color: Tuple[int, int, int],
        callback=None,
        scale: float = 1.0,
        thickness: int = 2,
    ) -> None:
        # (x, y) = top-left, (x2, y2) = bottom-right
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h
        self.label = label
        self.color = color
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = scale
        self.thickness = thickness
        self.callback = callback

    def draw(self, img) -> None:
        cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.color, -1)
        (text_width, text_height), _ = cv2.getTextSize(
            self.label, self.font, self.scale, self.thickness
        )
        x = self.x + (self.w - text_width) / 2
        y = self.y2 - (self.h - text_height) / 2  # align text base
        draw_text(img, x, y, self.label, _WHITE)

    def is_clicked(self, x: int, y: int) -> bool:
        return not (x < self.x or x > self.x2 or y < self.y or y > self.y2)


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
        self.idx_hand = 0
        self.auto_view = False
        self.mouse_set = False
        self.buttons = {}
        self._draw_msg_countdown = 0
        # self.test_df()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs["img"]
        width, height = img.shape[1], img.shape[0]

        if not self.mouse_set:
            cv2.setMouseCallback("PeekingDuck", mouse_click, self)
            self.mouse_set = True
            self.setup_gui(width, height)

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
        elif key == ord("#"):
            res = read_hand_catalog(self.conn)
            self._draw_msg(10, height - 60, f"DB: {res}", countdown=60)

        if idx_new is not None:
            self.idx_hand = idx_new % self.num_hands

        norm_landmarks, gesture_id = self.get_landmarks(self.idx_hand)
        hand_landmarks_2d = self.convert_norm_landmarks_to_2d_coords(
            norm_landmarks, num_pixels=500, base_coord=(int(width / 2), 600)
        )
        # print(hand_landmarks_2d)

        # render image
        blank_img = np.zeros((height, width, 3), dtype=np.uint8)
        self.draw_gui(blank_img)

        draw_text(blank_img, 10, 30, f"{HANDS_DB} Viewer")

        draw_hand(blank_img, hand_landmarks_2d)

        info_text = (
            f"{self.idx_hand + 1} of {self.num_hands}, gesture_id={gesture_id}"
            f" {'[auto]' if self.auto_view else ''}"
        )
        draw_text(blank_img, 400, 30, info_text)

        self._do_draw_msg(blank_img)

        outputs = {"img": blank_img}
        return outputs

    def draw_gui(self, img) -> None:
        for _, btn in self.buttons.items():
            btn.draw(img)

    def setup_gui(self, width: int, height: int) -> None:
        btn_prev_10x = MyButton(
            width - 5 * (10 + BTN_WIDTH),
            height - (10 + BTN_HEIGHT),
            BTN_WIDTH,
            BTN_HEIGHT,
            "<-",
            _BLUE,
            self.btn_prev_10x_cb,
        )
        btn_prev = MyButton(
            width - 4 * (10 + BTN_WIDTH),
            height - (10 + BTN_HEIGHT),
            BTN_WIDTH,
            BTN_HEIGHT,
            "<-",
            _BLUE,
            self.btn_prev_cb,
        )
        btn_next = MyButton(
            width - 3 * (10 + BTN_WIDTH),
            height - (10 + BTN_HEIGHT),
            BTN_WIDTH,
            BTN_HEIGHT,
            "->",
            _BLUE,
            self.btn_next_cb,
        )
        btn_next_10x = MyButton(
            width - 2 * (10 + BTN_WIDTH),
            height - (10 + BTN_HEIGHT),
            BTN_WIDTH,
            BTN_HEIGHT,
            "->",
            _BLUE,
            self.btn_next_10x_cb,
        )
        btn_auto = MyButton(
            width - (10 + BTN_WIDTH),
            height - (10 + BTN_HEIGHT),
            BTN_WIDTH,
            BTN_HEIGHT,
            "Auto",
            _BLUE,
            self.btn_auto_cb,
        )
        self.buttons = {
            "auto": btn_auto,
            "prev": btn_prev,
            "next": btn_next,
            "prev10x": btn_prev_10x,
            "next10x": btn_next_10x,
        }

    # Button callbacks
    def btn_prev_cb(self) -> None:
        idx_new = self.idx_hand - 1  # prev hand
        self.idx_hand = idx_new % self.num_hands

    def btn_next_cb(self) -> None:
        idx_new = self.idx_hand + 1  # next hand
        self.idx_hand = idx_new % self.num_hands

    def btn_auto_cb(self) -> None:
        self.auto_view = not self.auto_view

    def btn_prev_10x_cb(self) -> None:
        idx_new = self.idx_hand - 10  # prev hand
        self.idx_hand = idx_new % self.num_hands

    def btn_next_10x_cb(self) -> None:
        idx_new = self.idx_hand + 10  # next hand
        self.idx_hand = idx_new % self.num_hands

    # Object methods
    def get_landmarks(self, i: int) -> List[Any]:
        assert i >= 0
        df = self.df.iloc[[i]]
        gesture_id = df["gesture_id"].values[0]
        values = df.drop(["id", "gesture_id"], axis=1).values.flatten().tolist()
        landmarks = list(zip(values[::2], values[1::2]))
        return landmarks, gesture_id

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

    def test_df(self) -> None:
        """Test draw one hand landmark"""
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

    def _draw_msg(
        self,
        x: int,
        y: int,
        text: str,
        bgr: tuple = (255, 255, 255),
        scale: float = 1.0,
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
            scale=scale,
            thickness=thickness,
        )

    def _do_draw_msg(self, img):
        if self._draw_msg_countdown > 0:
            self._draw_msg_func(img)
            self._draw_msg_countdown -= 1
