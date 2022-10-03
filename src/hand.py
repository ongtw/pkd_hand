# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Encapsulate hand methods and utility functions.
"""

from typing import Any, List, Tuple
import copy
import itertools
import cv2
import numpy as np
import pandas as pd
import sqlite3

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

# Indices of The 21 Hand Landmarks
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# COLORS
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (128, 128, 128)
_RED = (0, 0, 255)
_GREEN = (0, 255, 0)
_BLUE = (255, 0, 0)
_YELLOW = (0, 255, 255)
_ORANGE = (0, 140, 238)
_PURPLE = (128, 0, 128)


####################
#
# Database Routines
#
####################
HANDS_DB = "hands.db"  # hands database file name


def connect_db():
    conn = None
    try:
        conn = sqlite3.connect(HANDS_DB)
    except sqlite3.Error as e:
        print(f"SQL error: {e}")
    return conn


def read_hands(conn) -> None:
    df = pd.read_sql("select * from hands order by id", conn)
    return df


def read_hand_catalog(conn) -> str:
    cur = conn.cursor()
    texts = []
    for row in cur.execute(
        "select gesture_id, count(*) from hands group by gesture_id"
    ):
        texts.append(f"{row[0]}:{row[1]}")
    res = " ".join(texts)
    # print(res)
    return res


def save_hand(conn, landmarks_2d, label: int) -> None:
    norm_landmarks = normalize_hand_landmarks(landmarks_2d)
    # Note: saving 43 data: 1 label + 21 x 2 = 42 hand coordinates
    sqlstr = (
        "insert into hands (gesture_id, x_0, y_0, "
        "x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, "
        "x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9, x_10, y_10, "
        "x_11, y_11, x_12, y_12, x_13, y_13, x_14, y_14, x_15, y_15, "
        "x_16, y_16, x_17, y_17, x_18, y_18, x_19, y_19, x_20, y_20) "
        f"values ({label}, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    cur = conn.cursor()
    cur.execute(sqlstr, norm_landmarks)
    conn.commit()


####################
#
# Display Routines
#
####################
def Point(x: float, y: float) -> Tuple[int, int]:
    return (int(x), int(y))


def draw_text(
    img,
    x: int,
    y: int,
    text: str,
    bgr: tuple = (255, 255, 255),
    size: float = 1.0,
    thickness: int = 2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    cv2.putText(img, text, Point(x, y), font, size, bgr, thickness, line)


####################
#
# Hand Routines
#
####################
def get_hand(results, label: str) -> object:
    """Return hand landmarks of 'label' hand

    Args:
        results (_type_): Mediapipe hands.process() results
        label (str): either "Left" or "Right" hand
    """
    if results.multi_hand_landmarks:
        handedness = results.multi_handedness
        # print("**********")
        # print(type(handedness))
        # print(dir(handedness))

        for i, (hand_landmarks, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # print(f"{i} {handedness.classification[0].label}")
            if handedness.classification[0].label == label:
                hand_landmarks = results.multi_hand_landmarks[i]
                return hand_landmarks

    return None


def calc_bounding_rect(image, landmarks) -> List[int]:
    """Calculate bounding box for hand wrt given image

    Args:
        image (np.array): Image data
        landmarks (mp.landmarks): Mediapipe hand landmarks data structure

    Returns:
        List[int]: List of topleft_x, topleft_y, bottomright_x, bottomright_y
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def project_hands_on_image(image, landmarks) -> List[Any]:
    """Project 3D hand landmarks onto 2D image

    Args:
        image (np.array): Image data
        landmarks (mp.landmarks): Mediapipe hand landmarks data structure

    Returns:
        List[Any]: List of 2D hand keypoints
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = []

    # Get key points
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points


def normalize_hand_landmarks(landmark_list) -> List[Any]:
    """Convert hand landmarks absolute coordinates (image coords) into relative
    coordinates in the range (-1, 1).
    The wrist is (0, 0).
    Negative values are to the left and above.
    Positive values are to the right and below.

    Args:
        landmark_list (mp.landmarks): MediaPipe hand landmarks data structure

    Returns:
        List[Any]: List of normalized hand landmarks.
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # print("temp_landmark_list:")
    # print(temp_landmark_list)

    # Convert to 1D list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    # print(f"max_value={max_value}")

    def normalize_(n):
        return n / max_value

    norm_landmark_list = list(map(normalize_, temp_landmark_list))

    # print("landmark_list:")
    # print(landmark_list)
    # print("norm_landmark_list:")
    # print(norm_landmark_list)

    return norm_landmark_list


def draw_hand(image, landmark_points):
    """Draw 2D hand landmark points in black and white

    Args:
        image (np.array): Image data
        landmark_points (mp.landmarks): Mediapipe hand landmarks data structure
    """
    if len(landmark_points) == 0:
        return

    col_palm = _GRAY
    col_fin_thumb = _RED
    col_fin_index = _PURPLE
    col_fin_middle = _GREEN
    col_fin_ring = _ORANGE
    col_fin_pinky = _BLUE

    # fmt: off
    # skeletal structure
    outline_color = _WHITE
    # thumb
    _color = col_fin_thumb
    cv2.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]), _color, 2)
    cv2.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]), _color, 2)

    # forefinger
    _color = col_fin_index
    cv2.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]), _color, 2)
    cv2.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]), _color, 2)
    cv2.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]), _color, 2)

    # middle finger
    _color = col_fin_middle
    cv2.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]), _color, 2)
    cv2.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]), _color, 2)
    cv2.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]), _color, 2)

    # ring finger
    _color = col_fin_ring
    cv2.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]), _color, 2)
    cv2.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]), _color, 2)
    cv2.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]), _color, 2)

    # pinky
    _color = col_fin_pinky
    cv2.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]), _color, 2)
    cv2.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]), _color, 2)
    cv2.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]), _color, 2)

    # palm
    _color = col_palm
    cv2.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]), _color, 2)
    cv2.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]), _color, 2)
    cv2.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]), _color, 2)
    cv2.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]), _color, 2)
    cv2.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]), _color, 2)
    cv2.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]), _color, 2)
    cv2.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]), outline_color, 6)
    cv2.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]), _color, 2)

    # keypoints
    outline_color = _WHITE
    big_dot = 8
    small_dot = 5
    for index, landmark in enumerate(landmark_points):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_thumb, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, col_fin_thumb, -1)
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, outline_color, 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_index, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_index, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, col_fin_index, -1)
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, outline_color, 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_middle, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_middle, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, col_fin_middle, -1)
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, outline_color, 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_ring, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_ring, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, col_fin_ring, -1)
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, outline_color, 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_palm, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_pinky, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, col_fin_pinky, -1)
            cv2.circle(image, (landmark[0], landmark[1]), small_dot, outline_color, 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, col_fin_pinky, -1)
            cv2.circle(image, (landmark[0], landmark[1]), big_dot, outline_color, 1)
    # fmt: on
