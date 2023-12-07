import keras.backend as K
import tensorflow as tf


def focal(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

        # compute the normalizer: the number of positive anchors
        normalizer = K.cast(K.shape(y_pred)[1], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(cls_loss) / normalizer

    return _focal


def iou():
    def _iou(y_true, y_pred):
        y_true = tf.maximum(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        # (num_pos, )
        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)
        w_expand = tf.maximum(pred_left, target_left)+tf.maximum(pred_right, target_right)
        h_expand = tf.maximum(pred_bottom, target_bottom)+tf.maximum(pred_top, target_top)
        are_expand = w_expand * h_expand
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        giou = (area_intersect + 1e-7) / (area_union + 1e-7)-(are_expand-area_union + 1e-7)/(are_expand + 1e-7)
        giou_loss = 1 - giou
        giou_loss = tf.reshape(giou_loss, [-1])

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(y_true)[1])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        return K.sum(giou_loss) / normalizer

    return _iou


def focal_with_mask(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(inputs):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
            cls_mask: (B, N)
            cls_num_pos: (B, )

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        y_true, y_pred, cls_mask, cls_num_pos = inputs[0], inputs[1], inputs[2], inputs[3]
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        # (B, N) --> (B, N, 1)
        cls_mask = tf.cast(cls_mask, tf.float32)
        cls_mask = tf.expand_dims(cls_mask, axis=-1)
        # (B, N, num_classes) * (B, N, 1)
        masked_cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred) * cls_mask
        # compute the normalizer: the number of positive locations
        normalizer = K.maximum(K.cast_to_floatx(1.0), tf.reduce_sum(cls_num_pos))
        return K.sum(masked_cls_loss) / normalizer

    return _focal


def iou_with_mask():
    def _iou(inputs):
        """

        Args:
            inputs: y_true: (B, N, 4) y_pred: (B, N, 4) regr_mask: (B, N)

        Returns:

        """
        y_true, y_pred, regr_mask = inputs[0], inputs[1], inputs[2]
        y_true = tf.maximum(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        # (B, N)
        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        # (B, N)
        target_area = (target_left + target_right) * (target_top + target_bottom)
        masked_target_area = tf.boolean_mask(target_area, regr_mask)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        masked_pred_area = tf.boolean_mask(pred_area, regr_mask)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        masked_area_intersect = tf.boolean_mask(area_intersect, regr_mask)
        masked_area_union = masked_target_area + masked_pred_area - masked_area_intersect

        w_expand = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        h_expand = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        area_expand = w_expand * h_expand
        masked_area_expand = tf.boolean_mask(area_expand, regr_mask)
        mask_giou = (masked_area_intersect + 1e-7) / (masked_area_union + 1e-7)-(masked_area_expand-masked_area_union + 1e-7)/(masked_area_expand + 1e-7)
        mask_giou_loss = 1 - mask_giou
        mask_giou_loss = tf.reshape(mask_giou_loss, [-1])

        # (B, N)
        regr_mask = tf.cast(regr_mask, tf.float32)
        # compute the normalizer: the number of positive locations
        regr_num_pos = tf.reduce_sum(regr_mask)
        normalizer = K.maximum(K.cast_to_floatx(1.), regr_num_pos)
        return K.sum(mask_giou_loss) / normalizer
    return _iou


def center_with_mask():
    def _centerness(inputs):
        # y_true-->[B,N]
        # regr_mask-->[B,N]
        y_true, y_pred, center_mask = inputs[0], inputs[1], inputs[2]
        y_true = tf.maximum(y_true, 0)
        masked_target_area = tf.boolean_mask(y_true, center_mask)
        masked_target_area = tf.reshape(masked_target_area, [-1])
        if tf.size(masked_target_area) == 0:
            loss = tf.constant(0.0)
            return loss
        # (B, N) --> (B, N, 1)
        center_mask = tf.cast(center_mask, tf.float32)
        center_mask = tf.expand_dims(center_mask, axis=-1)
        masked_cls_loss = K.binary_crossentropy(y_true, y_pred) * center_mask
        # compute the normalizer: the number of positive location
        normalizer = K.maximum(K.cast_to_floatx(1.0), tf.reduce_sum(center_mask))
        return K.sum(masked_cls_loss) / normalizer
    return _centerness


