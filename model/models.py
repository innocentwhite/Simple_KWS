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
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.
# Added new model definitions for speech command recognition used in
# the paper: https://arxiv.org/pdf/1711.07128.pdf
#
#

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""Model definitions for simple speech recognition.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer, out_dim,):
        super().__init__()
        self.gruLayer = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=hidden_layer, bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax()
        nn.init.xavier_uniform_(self.fcLayer.weight)
        nn.init.constant(self.fcLayer.bias, 0)

    def forward(self, x):
        out, h_n = self.gruLayer(x)# output shape is (batchsize, seq, features), h_n shape is (num_layers * num_directions,batchsize, features)
        h_n = h_n[-1]
        output = self.fcLayer(h_n)
        output = self.softmax(output)
        return output

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    config:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }

def create_model(config, model_settings):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    config:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        model_architecture: String specifying which kind of model to create.
        is_training: Whether the model is going to be used for training.
        runtime_settings: Dictionary of information about the runtime.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

    Raises:
        Exception: If the architecture type isn't recognized.
    """
    if config.arch =='gru':
        num_classes = model_settings['label_count']
        input_dim = model_settings['dct_coefficient_count']
        gru_units = config.model_size_info[1]
        layer_dim = config.model_size_info[0]
        return GRUModel(input_dim, gru_units, layer_dim, num_classes)
    else:
        raise Exception('model_architecture argument "' + config.arch +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv", "low_latency_svdf",'+
                        ' "dnn", "cnn", "basic_lstm", "lstm",'+
                        ' "gru", "crnn" or "ds_cnn"')