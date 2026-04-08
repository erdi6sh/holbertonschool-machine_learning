#!/usr/bin/env python3
"""Module for YOLO v3 object detection using Darknet Keras model."""
import tensorflow as tf


class Yolo:
    """Class that uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detection model.

        Args:
            model_path: path to the Darknet Keras model (.h5 file)
            classes_path: path to the file containing class names
            class_t: float, box score threshold for initial filtering
            nms_t: float, IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all anchor boxes
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
