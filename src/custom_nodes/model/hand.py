from tkinter import W
import mediapipe as mp


"""The 21 hand landmarks."""
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

        for i, hand_class_list in enumerate(handedness):
            # print(type(hand_class_list), hand_class_list)
            # print(dir(hand_class_list))

            hand_container = hand_class_list.classification
            # print(type(hand_container), hand_container)
            # print(dir(hand_container))

            hand_classification = hand_container[0]
            # print(type(hand_classification), hand_classification)
            # print(dir(hand_classification))

            if hand_classification.label == label:
                # object = NormalizedLandmarkList
                hand_landmarks = results.multi_hand_landmarks[i]

                # object = LandmarkList
                # hand_landmarks = results.multi_hand_world_landmarks[i]

                return hand_landmarks

    return None


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
