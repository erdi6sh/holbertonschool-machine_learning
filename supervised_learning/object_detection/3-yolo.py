def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
    """Apply non-max suppression to eliminate redundant overlapping boxes.

    Args:
        filtered_boxes: numpy.ndarray of shape (?, 4) containing all
            filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class
            number for each filtered box
        box_scores: numpy.ndarray of shape (?) containing the box scores
            for each filtered box

    Returns:
        tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
            box_predictions: numpy.ndarray of shape (?, 4)
            predicted_box_classes: numpy.ndarray of shape (?,)
            predicted_box_scores: numpy.ndarray of shape (?)
    """
    box_predictions = []
    predicted_box_classes = []
    predicted_box_scores = []

    unique_classes = np.unique(box_classes)

    for cls in unique_classes:
        cls_mask = box_classes == cls
        cls_boxes = filtered_boxes[cls_mask]
        cls_scores = box_scores[cls_mask]

        order = np.argsort(cls_scores)[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        keep = []
        while len(cls_boxes) > 0:
            keep.append(0)
            if len(cls_boxes) == 1:
                break
            ious = iou(cls_boxes[0], cls_boxes[1:])
            suppress = np.where(ious > self.nms_t)[0]
            keep_mask = np.ones(len(cls_boxes) - 1, dtype=bool)
            keep_mask[suppress] = False
            cls_boxes = cls_boxes[1:][keep_mask]
            cls_scores = cls_scores[1:][keep_mask]

        box_predictions.append(cls_boxes[:len(keep)])
        predicted_box_classes.append(
            np.full(len(keep), cls, dtype=box_classes.dtype))
        predicted_box_scores.append(cls_scores[:len(keep)])

    box_predictions = np.concatenate(box_predictions, axis=0)
    predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
    predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

    return box_predictions, predicted_box_classes, predicted_box_scores
