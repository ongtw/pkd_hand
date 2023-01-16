# pkd_hands
### By David Ong Tat-Wee, Oct 2022


Introduction
============


This is a set of software to demonstrate how to use MediaPipe's Hands model within
PeekingDuck.

The components are:

* `src/custom_nodes/dabble/hand_capture.py`: PeekingDuck custom node to show live hand
and to capture landmark coordinates into SQL database.

* `src/custom_nodes/dabble/hand_view.py`: PeekingDuck custom node to show rendered
hand using data within SQL database.

* `src/custom_nodes/model/hand_inference.py`: PeekingDuck custom node to perform
inference on hand(s) and recognise hand gestures.

* `src/hand.py`: encapsulates all hand related functions and methods.

* `src/train_hand_gestures_classifier.py`: script to train a fully connected network
for hand gestures classification under Tensorflow.



Run
===

Run the custom nodes with `peekingduck run --config_path <pipeline yaml>`

E.g. `peekingduck run --config_path inference_hands.yml` will run hand inference
using PeekingDuck's `custom_nodes.dabble.hand_inference` node.


Pipeline Files
--------------
`capture_hands.yml` <br>
`inference_hands.yml` <br>
`view_hands.yml`



Jira Integration
================

New note to test Jira integration.
