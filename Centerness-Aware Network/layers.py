import keras
import keras.backend as K
import tensorflow as tf
from utils import anchors as utils_anchors
import util_graphs
from keras.layers import Conv2D
from deform_conv import tf_batch_map_offsets
from util_graphs import prop_box_graph_2
import numpy as np


class Anchors(keras.layers.Layer):
    """
    Keras layer for generating anchors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """
        Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = utils_anchors.AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)
        if scales is None:
            self.scales = utils_anchors.AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors = K.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        feature = inputs
        feature_shape = K.shape(feature)

        # generate proposals from bbox deltas and shifted anchors
        if K.image_data_format() == 'channels_first':
            anchors = util_graphs.shift(feature_shape[2:4], self.stride, self.anchors)
        else:
            # (fh * fw * num_anchors, 4)
            anchors = util_graphs.shift(feature_shape[1:3], self.stride, self.anchors)
        # (b, fh * fw * num_anchors, 4)
        anchors = K.tile(K.expand_dims(anchors, axis=0), (feature_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if K.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return input_shape[0], total, 4
        else:
            return input_shape[0], None, 4

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config


class UpsampleLike(keras.layers.Layer):
    """
    Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        if K.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = util_graphs.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return util_graphs.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class ClipBoxes(keras.layers.Layer):
    """
    Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

        return K.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def filter_detections(
        boxes,
        classification,
        centerness,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.05,
        max_detections=300,
        nms_threshold=0.5
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other: List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(K.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            filtered_scores = K.gather(scores_, indices_)[:, 0]
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                      iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = K.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = K.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((K.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = K.concatenate(all_indices, axis=0)
    else:
        scores = K.max(classification, axis=1)
        labels = K.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    # classification = classification * centerness
    # classification = K.sqrt(classification)
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=K.minimum(max_detections, K.shape(scores)[0]))

    # filter input using the final set of indices
    indices = K.gather(indices[:, 0], top_indices)
    boxes = K.gather(boxes, indices)
    labels = K.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = K.maximum(0, max_detections - K.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = K.cast(labels, 'int32')


    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        centerness = inputs[2]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            centerness_ = args[2]

            return filter_detections(
                boxes_,
                classification_,
                centerness_,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch item
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, centerness],
            dtype=[K.floatx(), K.floatx(), 'int32'],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
                   (input_shape[0][0], self.max_detections, 4),
                   (input_shape[1][0], self.max_detections),
                   (input_shape[1][0], self.max_detections),
               ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config


class ConvOffset2D(Conv2D):  # 继承2D卷积
    """ConvOffset2D"""

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init"""

        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,  # 由于要计算x,y坐标的偏移量，所以需要两倍的channels
            # TODO gradients are near zero if init is zeros
            kernel_initializer='zeros',
            # kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x):
        # TODO offsets probably have no nonlinearity?
        x_shape = x.get_shape()  # 输入tensor的shape=(b,h,w,c)
        offsets = super(ConvOffset2D, self).call(x)  # 进行对输入卷积，2*channels,shape=(b,h,w,2c)
        offsets = self._to_bc_h_w_2(offsets, x_shape)  # 将offses的shape转化为(bc,h,w,2),两个通道分别表示x,y的偏移量
        x = self._to_bc_h_w(x, x_shape)  # 将输入shape变为(bc,h,w)
        x_offset = tf_batch_map_offsets(x, offsets)  # 得到片以后新坐标的所有像素值
        x_offset = self._to_b_h_w_c(x_offset, x_shape)  # 变换维度
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])  # 交换维度(b,2c,h,w)
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))  # (bc,h,w,2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])  # 交换维度(b,c,h,w)
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))  # (bc,h,w)
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


def build_centerness_target(gt_box_levels, gt_boxes, feature_shapes, strides):
    ignore_scale = 0.5
    gt_boxes = gt_boxes[:, :4]
    center_target = tf.zeros((0, 1))
    center_mask = tf.zeros((0,), dtype=tf.bool)
    for level_id in range(len(strides)):
        feature_shape = feature_shapes[level_id]
        stride = strides[level_id]
        fh = feature_shape[0]
        fw = feature_shape[1]
        level_gt_box_indices = tf.where(tf.equal(gt_box_levels, level_id))
        def do_level_has_gt_boxes():
            level_gt_boxes = tf.gather(gt_boxes, level_gt_box_indices[:, 0])
            level_proj_boxes = level_gt_boxes / stride
            ign_x1, ign_y1, ign_x2, ign_y2 = prop_box_graph_2(level_proj_boxes, ignore_scale, fw, fh)
            # pos_x1, pos_y1, pos_x2, pos_y2 = prop_box_graph_2(level_proj_boxes, pos_scale, fw, fh)
            def build_single_gt_box_cnterness_target(args):
                ign_x1_ = args[0]
                ign_y1_ = args[1]
                ign_x2_ = args[2]
                ign_y2_ = args[3]
                gt_box  = args[4]
                # pos_x1_ = args[4]
                # pos_y1_ = args[5]
                # pos_x2_ = args[6]
                # pos_y2_ = args[7]
                # gt_box = args[8]
                # ign_top_bot = tf.concat((pos_y1_ - ign_y1_, ign_y2_ - pos_y2_), axis=0)
                # ign_lef_rit = tf.concat((pos_x1_ - ign_x1_, ign_x2_ - pos_x2_), axis=0)
                # ign_pad = tf.stack([ign_top_bot, ign_lef_rit], axis=0)
                other_top_bot = tf.concat((ign_y1_, fh - ign_y2_), axis=0)
                other_lef_rit = tf.concat((ign_x1_, fw - ign_x2_), axis=0)
                other_pad = tf.stack([other_top_bot, other_lef_rit], axis=0)
                locs_x = K.arange(ign_x1_[0], ign_x2_[0], dtype=tf.float32)
                locs_y = K.arange(ign_y1_[0], ign_y2_[0], dtype=tf.float32)
                shift_x = (locs_x + 0.5) * stride
                shift_y = (locs_y + 0.5) * stride
                shift_xx, shift_yy = tf.meshgrid(shift_x, shift_y)
                shifts = K.stack((shift_xx, shift_yy, shift_xx, shift_yy), axis=-1)
                l = shifts[:, :, 0] - gt_box[0]
                t = shifts[:, :, 1] - gt_box[1]
                r = gt_box[2] - shifts[:, :, 2]
                b = gt_box[3] - shifts[:, :, 3]
                level_box_pos_area = (l + r) * (t + b)
                # level_box_area = tf.pad(level_box_pos_area, ign_pad + other_pad, constant_values=1e7)
                level_box_area = tf.pad(level_box_pos_area, other_pad, constant_values=1e7)
                level_box_center_pos_target = tf.sqrt(tf.abs((tf.minimum(l, t) * tf.minimum(t, b)) / (tf.maximum(l, r) * tf.maximum(t, b))))
                level_box_center_pos_target = tf.expand_dims(level_box_center_pos_target, axis=-1)
                # level_box_center_target = tf.pad(level_box_center_pos_target,
                #                                tf.concat((ign_pad + other_pad, tf.constant([[0, 0]])), axis=0))
                level_box_center_target = tf.pad(level_box_center_pos_target,
                                               tf.concat((other_pad, tf.constant([[0, 0]])), axis=0))
                level_box_center_pos_mask = tf.ones((ign_y2_[0] - ign_y1_[0], ign_x2_[0] - ign_x1_[0]))
                level_box_center_mask = tf.pad(level_box_center_pos_mask, other_pad)
                return level_box_area, level_box_center_target, level_box_center_mask
            level_area, level_center_target, level_center_mask = tf.map_fn(
                build_single_gt_box_cnterness_target,
                elems=[
                    ign_x1, ign_y1, ign_x2, ign_y2,level_gt_boxes
                    ],
                dtype=(tf.float32, tf.float32, tf.float32)
            )
            level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
            level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))
            locs_x = K.arange(0, fw)
            locs_y = K.arange(0, fh)
            locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
            locs_xx = tf.reshape(locs_xx, (-1,))
            locs_yy = tf.reshape(locs_yy, (-1,))
            level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)
            level_center_target_ = tf.gather_nd(level_center_target, level_indices)
            level_center_mask = tf.reduce_sum(level_center_mask, axis=0) > 0
            level_center_mask_ = tf.reshape(level_center_mask, (fh * fw,))
            return level_center_target_, level_center_mask_
        def do_level_has_no_gt_boxes():
            level_center_target_ = tf.zeros((fh * fw, 1))
            level_regr_mask_ = tf.zeros((fh * fw,), dtype=tf.bool)
            return level_center_target_, level_regr_mask_
        level_center_target, level_centerness_mask = tf.cond(
            tf.equal(tf.size(level_gt_box_indices), 0),
            do_level_has_no_gt_boxes,
            do_level_has_gt_boxes)
        center_target = tf.concat([center_target, level_center_target], axis=0)
        center_mask = tf.concat([center_mask, level_centerness_mask], axis=0)
    return center_target, center_mask
