# End-to-End Framework for Ergonomic Risk Assessment for Electric Line Workers

In this repository you can find the codes for the end-to-end framework for ergonomic risk assessment for electric line workers using acceleration signals from a wrist-worn accelerometer. The framework steps include segmentation, activity classification, and activity repetition counting. A animation of the steps is presented below.



![Alt Text](./readme-mtls/workflow.gif)


The activities in this data set include:

1. Sitting on a chair while keeping hands still on the chair arms for 3 minutes
2. Standing still for 3 minutes
3. Walking on a set path for 3 minutes
4. Hoisting a weighted bucket up and down to a height of 4 m for 10 repetitions
5. Lifting and lowering a weighted box for 20 repetitions
6. Pushing a cart on a set path for 10 repetitions
7. Typing on a computer for 3 minutes
8. Climbing up and down a ladder for 20 repetitions
9. Working on an electrical panel for 3 minutes
10. Inserting screws using a screw driver at an overhead height for 3 minute


The one-time classifier training is performed and the trained model is saved in [trained_model/tf_model](trained_model/tf_model).

