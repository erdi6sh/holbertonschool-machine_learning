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
        """Process the outputs from the Darknet model for a single image.

        Args:
            outputs: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                containing predictions from the Darknet model
            image_size: numpy.ndarray containing the image's original size
                [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
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

            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height
            b_w = (p_w * np.exp(t_w)) / input_width
            b_h = (p_h * np.exp(t_h)) / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            box_confidence = sigmoid(output[:, :, :, 4:5])
            box_confidences.append(box_confidence)

            box_class_prob = sigmoid(output[:, :, :, 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes based on box score threshold.

        Args:
            boxes: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            box_confidences: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_probs
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores


def sigmoid(x):
    """Calculate the sigmoid of a numpy array."""
    return 1 / (1 + np.exp(-x))
