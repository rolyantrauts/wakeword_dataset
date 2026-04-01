# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Depthwise separable convolution neural network."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils

def model_parameters(parser_nn):
  """Depthwise separable convolution neural network model parameters."""
  parser_nn.add_argument(
      '--cnn1_kernel_size',
      type=str,
      default='(10,4)',
      help='Height and width of the 2D convolution window',
  )
  parser_nn.add_argument(
      '--cnn1_dilation_rate',
      type=str,
      default='(1,1)',
      help='Dilation rate to use for dilated convolutions',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='(2,2)',
      help='Strides of the convolution along the height and width',
  )
  parser_nn.add_argument(
      '--cnn1_padding',
      type=str,
      default='same',
      help='Padding method',
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=int,
      default=64,
      help='Number of output filters in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn1_act',
      type=str,
      default='relu',
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--bn_momentum',
      type=float,
      default=0.98,
      help='Momentum for the moving average',
  )
  parser_nn.add_argument(
      '--bn_center',
      type=int,
      default=1,
      help='If True, add offset of beta to normalized tensor.'
      'If False, beta is ignored',
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=0,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), '
      'this can be disabled since the scaling will be done by the next layer.',
  )
  parser_nn.add_argument(
      '--bn_renorm',
      type=int,
      default=0,
      help='Whether to use Batch Renormalization',
  )
  parser_nn.add_argument(
      '--dw2_kernel_size',
      type=str,
      default='(3,3),(3,3),(3,3),(3,3),(3,3)',
      help='Height and width of the 2D convolution window',
  )
  parser_nn.add_argument(
      '--dw2_dilation_rate',
      type=str,
      default='(1,1),(1,1),(1,1),(1,1),(1,1)',
      help='Dilation rate to use for dilated convolutions',
  )
  parser_nn.add_argument(
      '--dw2_strides',
      type=str,
      default='(1,1),(1,1),(1,1),(1,1),(1,1)',
      help='Strides of the convolution along the height and width',
  )
  parser_nn.add_argument(
      '--dw2_padding',
      type=str,
      default="'same','same','same','same','same'",
      help='Padding method',
  )
  parser_nn.add_argument(
      '--dw2_act',
      type=str,
      default="'relu','relu','relu','relu','relu'",
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn2_filters',
      type=str,
      default='64,64,64,64,128',
      help='Number of output filters in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn2_act',
      type=str,
      default="'relu','relu','relu','relu','relu'",
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.2,
      help='Percentage of data dropped in final dense layer',
  )
  parser_nn.add_argument(
      '--dropout_cnn',
      type=float,
      default=0.0,
      help='Percentage of data dropped after convolutional blocks',
  )


def model(flags):
  """Depthwise separable convolution neural network (DS CNN) model.

  It is based on paper:
  Hello Edge: Keyword Spotting on Microcontrollers
  https://arxiv.org/pdf/1711.07128.pdf

  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  net = tf.keras.backend.expand_dims(net)

  # Safely fetch the l2 weight decay flag and convert to a Keras regularizer
  l2_reg = tf.keras.regularizers.l2(flags.l2_weight_decay) if getattr(flags, 'l2_weight_decay', 0.0) > 0 else None
  dropout_cnn = getattr(flags, 'dropout_cnn', 0.0)
  
  # Determine if we are actively training to enforce dropout behavior
  is_training = bool(getattr(flags, 'train', 0))

  net = stream.Stream(
      cell=tf.keras.layers.Conv2D(
          filters=flags.cnn1_filters,
          kernel_size=utils.parse(flags.cnn1_kernel_size),
          strides=utils.parse(flags.cnn1_strides),
          padding=flags.cnn1_padding,
          dilation_rate=utils.parse(flags.cnn1_dilation_rate),
          kernel_regularizer=l2_reg))( # L2 applied here for standard conv
              net)

  net = tf.keras.layers.BatchNormalization(
      momentum=flags.bn_momentum,
      center=flags.bn_center,
      scale=flags.bn_scale,
      renorm=flags.bn_renorm)(net)
  net = tf.keras.layers.Activation(flags.cnn1_act)(net)
  
  if dropout_cnn > 0.0:
      net = tf.keras.layers.Dropout(rate=dropout_cnn)(net, training=is_training)

  # Handle padding by splitting the string instead of using utils.parse
  paddings = flags.dw2_padding.split(',')

  for i, (kernel_size, dilation_rate, strides, dw2_act, cnn2_filters, cnn2_act) in enumerate(zip(
      utils.parse(flags.dw2_kernel_size), utils.parse(flags.dw2_dilation_rate),
      utils.parse(flags.dw2_strides),
      utils.parse(flags.dw2_act), utils.parse(flags.cnn2_filters),
      utils.parse(flags.cnn2_act))):

    # Use the specific padding if provided in a list, otherwise default to 'same'
    padding = paddings[i].strip().strip("'").strip('"') if i < len(paddings) else 'same'

    net = stream.Stream(
        cell=tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            depthwise_regularizer=l2_reg))( # L2 applied to depthwise kernel
                net)

    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(net)
    net = tf.keras.layers.Activation(dw2_act)(net)

    net = tf.keras.layers.Conv2D(
        filters=cnn2_filters, 
        kernel_size=(1, 1), 
        strides=(1, 1),
        padding='valid',
        kernel_regularizer=l2_reg)(net) # L2 applied here for pointwise conv

    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(net)
    net = tf.keras.layers.Activation(cnn2_act)(net)
    
    if dropout_cnn > 0.0:
        net = tf.keras.layers.Dropout(rate=dropout_cnn)(net, training=is_training)

  net = tf.keras.layers.GlobalAveragePooling2D()(net)
  if flags.dropout1 > 0.0:
    net = tf.keras.layers.Dropout(rate=flags.dropout1)(net, training=is_training)

  net = tf.keras.layers.Dense(flags.label_count, kernel_regularizer=l2_reg)(net) # L2 applied to final layer

  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)

  return tf.keras.Model(input_audio, net)