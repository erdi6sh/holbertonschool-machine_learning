#!/usr/bin/env python3
"""Module for YOLO v3 object detection using Darknet Keras model."""
import numpy as np
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

    def process_outputs(self, outputs, image_size):
        """Process the outputs from the Darknet model.

        Args:
            outputs: list of numpy.ndarrays containing predictions from
                the Darknet model for a single image. Each output has shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing the image original size
                [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs):
                boxes: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 4) with
                    boundary boxes (x1, y1, x2, y2) relative to original image
                box_confidences: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 1)
                box_class_probs: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, classes)
        """
        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, (grid_height, 1, anchor_boxes))

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(c_y, (1, grid_width, anchor_boxes))

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height
            b_w = (p_w * np.exp(t_w)) / input_width
            b_h = (p_h * np.exp(t_h)) / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            box_confidence = 1 / (1 + np.exp(-output[:, :, :, 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[:, :, :, 5:]))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
