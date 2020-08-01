# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for ssd resnet v1 FPN feature extractors."""
import abc
import itertools
import numpy as np
import tensorflow as tf

from object_detection.models import ssd_feature_extractor_test


class SSDResnetFPNFeatureExtractorTestBase(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):
  """Helper test class for SSD Resnet v1 FPN feature extractors."""

  @abc.abstractmethod
  def _resnet_scope_name(self):
    pass

  @abc.abstractmethod
  def _fpn_scope_name(self):
    return 'fpn'

  @abc.abstractmethod
  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                min_depth=32):
    pass

  def test_extract_features_returns_correct_shapes_256(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_dynamic_inputs(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_depth_multiplier(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5
    expected_num_channels = int(256 * depth_multiplier)
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, expected_num_channels),
                                  (2, 16, 16, expected_num_channels),
                                  (2, 8, 8, expected_num_channels),
                                  (2, 4, 4, expected_num_channels),
                                  (2, 2, 2, expected_num_channels)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_min_depth(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    min_depth = 320
    expected_feature_map_shape = [(2, 32, 32, min_depth),
                                  (2, 16, 16, min_depth),
                                  (2, 8, 8, min_depth),
                                  (2, 4, 4, min_depth),
                                  (2, 2, 2, min_depth)]

    def graph_fn(image_tensor):
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple, min_depth=min_depth)
      return feature_extractor.extract_features(image_tensor)

    image_tensor = np.random.rand(2, image_height, image_width,
                                  3).astype(np.float32)
    feature_maps = self.execute(graph_fn, [image_tensor])
    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shape):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
    image_height = 254
    image_width = 254
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]

    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_raises_error_with_invalid_image_size(self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier, pad_to_multiple)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = tf.constant(np.random.rand(4, image_height, image_width, 3))
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(test_image)
    with self.test_session() as sess:
      test_image_out, preprocessed_image_out = sess.run(
          [test_image, preprocessed_image])
      self.assertAllClose(preprocessed_image_out,
                          test_image_out - [[123.68, 116.779, 103.939]])

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    pad_to_multiple = 1
    g = tf.Graph()
    with g.as_default():
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple)
      preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
      feature_extractor.extract_features(preprocessed_inputs)
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      for variable in variables:
        self.assertTrue(
            variable.name.startswith(self._resnet_scope_name())
            or variable.name.startswith(self._fpn_scope_name()))
