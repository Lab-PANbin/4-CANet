"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import initializers
import layers
from models import assert_training_model
from fsaf_layers import LevelSelect, CAMTarget, Locations, RegressBoxes, get_centermask_graph
from losses import focal_with_mask, iou_with_mask, center_with_mask
import configure
import tensorflow as tf


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """
    Creates the FPN layers on top of the backbone features.

    Args
        C3: Feature stage C3 from the backbone.
        C4: Feature stage C4 from the backbone.
        C5: Feature stage C5 from the backbone.
        feature_size: The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # "C6 is obtained via a 3x3 stride-2 conv on C5"
    C6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='C6')(C5)

    # "C7 is computed by applying ReLU followed by a 3x3 stride-2 conv on C6"
    C7 = keras.layers.Activation('relu', name='C6_relu')(C6)
    C7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='C7')(C7)

    # upsample C5 to get P5 from the FPN paper
    P7 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C7_reduced')(C7)
    P7_upsampled = layers.UpsampleLike(name='P7_upsampled')([P7, C6])
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P7')(P7)

    # add P7 elementwise to C6
    P6 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C6_reduced')(C6)
    P6 = keras.layers.Add(name='P6_merged')([P7_upsampled, P6])
    P6_upsampled = layers.UpsampleLike(name='P6_upsampled')([P6, C5])
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P6')(P6)

    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5 = keras.layers.Add(name='P5_merged')([P6_upsampled, P5])
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    return [P3, P4, P5, P6, P7]


def __create_classification_head(
        num_classes,
        pyramid_feature_size=256,
        classification_feature_size=256,
        prior_probability=0.01,
        name='CANet_classification_model'
):
    """
    Creates the default classification submodel.

    Args
        num_classes: Number of classes to predict a score for at each feature level.
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_shared_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='CANet_classification_head',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='CANet_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='CANet_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_regression_head(
        num_values,
        pyramid_feature_size=256,
        regression_feature_size=256,
        prior_probability=0.01,
        name='CANet_regression_model'
):
    """
    Creates the default regression submodel.

    Args
        num_values: Number of values to regress.
        num_anchors: Number of anchors to regress for each feature level.
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)
    # outputs = keras.layers.Conv2D(num_values, name='pyramid_regression', activation='relu', **options)(outputs)
    # # (b, h*w , num_values)
    # outputs = keras.layers.Reshape((-1, num_values), name='CANet_regression_reshape')(outputs)
    # outputs = keras.layers.Activation('relu', name='CANet_regression_relu')(outputs)

    outputs = keras.layers.Conv2D(num_values, name='pyramid_regression', **options)(outputs)
    # (b, num_anchors_this_feature_map, num_values)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)
    outputs = keras.layers.Lambda(lambda x: keras.backend.exp(x))(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_centerness_model(
        num_values,
        pyramid_feature_size=256,
        centerness_feature_size=256,
        prior_probability=0.01,
        name='CANet_centerness_model'
):
    """
    Creates the default regression submodel.

    Args
        num_values: Number of values to regress.
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same'
    }
    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=centerness_feature_size,
            activation='sigmoid',
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zero',
            name='pyramid_centerness_{}'.format(i),
            **options
        )(outputs)
    # outputs = keras.layers.Conv2D(num_values, name='pyramid_centerness', activation='sigmoid', **options)(outputs)
    outputs = keras.layers.Conv2D(
        filters=num_values,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        # bias_initializer='zero',
        name='CANet_centerness_head',
        **options
    )(outputs)

    # (b, h*w , num_values)
    outputs = keras.layers.Reshape((-1, num_values), name='CANet_centerness_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='CANet_centerness_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __construct_CAM_submodels(num_classes):
    """
    Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes: Number of classes to use.
        num_anchors: Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """

    return [
        ('CAM_centernessmask', __create_centerness_model(num_values=1)),
        ('CAM_regression', __create_regression_head(num_values=4)),
        ('CAM_classification', __create_classification_head(num_classes=num_classes))
    ]


def __construct_CAM_single_pyramid(model_output_name, model, features):
    """
    Applies a single submodel to each FPN level.

    Args
        name: Name of the submodel.
        model: The submodel to evaluate.
        features: The FPN features. [P3, P4, P5, P6, P7]

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """

    return keras.layers.Concatenate(axis=1, name=model_output_name)([model(f) for f in features])


def __construct_CAM_pyramid(models, features):
    """
    Applies all submodels to each FPN level.

    Args
        models: List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features: The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """

    return [__construct_CAM_single_pyramid(model_output_name, model, features) for model_output_name, model in models]

def __construct_MSCD(batch_centerness_pred, features):
    """
    Applies multiscale centerness descriptor (MSCD) to each FPN level.

    Args
        batch_centerness_pred
        features: The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    sum0 = tf.shape(features[0])[1] * tf.shape(features[0])[2]
    sum1 = sum0 + tf.shape(features[1])[1] * tf.shape(features[1])[2]
    sum2 = sum1 + tf.shape(features[2])[1] * tf.shape(features[2])[2]
    sum3 = sum2 + tf.shape(features[3])[1] * tf.shape(features[3])[2]
    features[0] = keras.layers.Lambda(get_centermask_graph, arguments={'begin_count': 0})([batch_centerness_pred, features[0]])
    features[1] = keras.layers.Lambda(get_centermask_graph, arguments={'begin_count': sum0})([batch_centerness_pred, features[1]])
    features[2] = keras.layers.Lambda(get_centermask_graph, arguments={'begin_count': sum1})([batch_centerness_pred, features[2]])
    features[3] = keras.layers.Lambda(get_centermask_graph, arguments={'begin_count': sum2})([batch_centerness_pred, features[3]])
    features[4] = keras.layers.Lambda(get_centermask_graph, arguments={'begin_count': sum3})([batch_centerness_pred, features[4]])
    return features

def CANet(
        inputs,
        backbone_layers,
        num_classes,
        name='CANet'
):
    """
    Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs: keras.layers.Input (or list of) for the input to the model.
        num_classes: Number of classes to classify.
        num_anchors: Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels: Submodels to run on each feature map (default is regression and classification submodels).
        name: Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    # inputs=[gt_boxes_input, feature_shapes_input]]
    gt_boxes_input = inputs[1]
    feature_shapes_input = inputs[2]
    # create_pyramid_features [P3, P4, P5, P6, P7].
    # generate  centerness heatmap for all pyramid levels.
    # [(b, sum(fh*fw), 4), (b, sum(fh*fw), num_classes),(b, sum(fh*fw), 1)]
    C3, C4, C5 = backbone_layers
    features = __create_pyramid_features(C3, C4, C5)
    submodels = __construct_CAM_submodels(num_classes)
    batch_centerness_pred = __construct_CAM_pyramid(submodels[0:1], features)[0]

    # execute sub-modules in Centerness-Aware Model
    features = __construct_MSCD(batch_centerness_pred, features)
    batch_regr_pred, batch_cls_pred = __construct_CAM_pyramid(submodels[1:], features)
    batch_gt_box_levels = LevelSelect(name='level_select')(
        [batch_cls_pred, batch_regr_pred, feature_shapes_input, gt_boxes_input])

    # generate hybrid loss function
    batch_cls_target, batch_cls_mask, batch_cls_num_pos, batch_regr_target, batch_regr_mask, batch_center_target, batch_center_mask = CAMTarget(
        num_classes=num_classes,
        name='CAM_target')(
        [batch_gt_box_levels, feature_shapes_input, gt_boxes_input])
    focal_loss_graph = focal_with_mask()
    iou_loss_graph = iou_with_mask()
    center_loss_graph = center_with_mask()
    cls_loss = keras.layers.Lambda(focal_loss_graph,
                                   output_shape=(1,),
                                   name="cls_loss")(
        [batch_cls_target, batch_cls_pred, batch_cls_mask, batch_cls_num_pos])
    regr_loss = keras.layers.Lambda(iou_loss_graph,
                                    output_shape=(1,),
                                    name="regr_loss")([batch_regr_target, batch_regr_pred, batch_regr_mask])
    center_loss = keras.layers.Lambda(center_loss_graph,
                                    output_shape=(1,),
                                    name="center_loss")([batch_center_target, batch_centerness_pred, batch_center_mask])
    return keras.models.Model(inputs=inputs,
                              outputs=[cls_loss, regr_loss, center_loss, batch_cls_pred, batch_regr_pred, batch_centerness_pred],
                              name=name)


def CAM_predict_box(
        model=None,
        nms=True,
        class_specific_filter=True,
        name='CAM_predict_box',
):
    """
    Construct a centerness-aware model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the centerness-aware model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model: RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms: Whether to use non-maximum suppression for the filtering step.
        class_specific_filter: Whether to use class specific filtering or filter for the best scoring class only.
        name: Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    # create RetinaNet model
    assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    # (b, sum(fh*fw), num_classes)
    classification = model.outputs[3]
    centerness = model.outputs[5]

    # (b, sum(fh*fw), 4)
    regression = model.outputs[4]
    locations, strides = Locations(strides=configure.STRIDES)(features)

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([locations, strides, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification, centerness])

    # construct the model
    return keras.models.Model(inputs=model.inputs[0], outputs=detections, name=name)
