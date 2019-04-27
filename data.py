# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for loading the automobile data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow as tf

URL = "/Users/agustinalbinati/Documents/developer/TensorFlow/Matias/Kaggle/ctr_data/test.csv"
URL = "/Users/agustinalbinati/Documents/developer/TensorFlow/Matias/Kaggle/ctr_data/ctr_15.csv"

# Order is important for the csv-readers, so we use an OrderedDict here.
COLUMN_TYPES = collections.OrderedDict([
    ("Label", int),
    ("action_categorical_0", str),
    ("action_categorical_1", str),
    ("action_categorical_2", str),
    ("action_categorical_3", str),
    ("action_categorical_4", str),
    ("action_categorical_5", str),
    ("action_categorical_6", str),
    ("action_categorical_7", str),
    ("action_list_0", str),
    ("action_list_1", str),
    ("action_list_2", str),
    ("auction_age", float),
    ("auction_bidfloor", float),
    ("auction_boolean_0", str),
    ("auction_boolean_1", str),
    ("auction_boolean_2", str),
    ("auction_categorical_0", str),
    ("auction_categorical_1", str),
    ("auction_categorical_10", str),
    ("auction_categorical_11", str),
    ("auction_categorical_12", str),
    ("auction_categorical_2", str),
    ("auction_categorical_3", str),
    ("auction_categorical_4", str),
    ("auction_categorical_5", str),
    ("auction_categorical_6", str),
    ("auction_categorical_7", str),
    ("auction_categorical_8", str),
    ("auction_categorical_9", str),
    ("auction_list_0", str),
    ("auction_time", str),
    ("creative_categorical_0", str),
    ("creative_categorical_1", str),
    ("creative_categorical_10", str),
    ("creative_categorical_11", str),
    ("creative_categorical_12", str),
    ("creative_categorical_2", str),
    ("creative_categorical_3", str),
    ("creative_categorical_4", str),
    ("creative_categorical_5", str),
    ("creative_categorical_6", str),
    ("creative_categorical_7", str),
    ("creative_categorical_8", str),
    ("creative_categorical_9", str),
    ("creative_height", float),
    ("creative_width", float),
    ("device_id", str),
    ("device_id_type", str),
    ("gender", str),
    ("has_video", str),
    ("timezone_offset", float),
])


def raw_dataframe():
  """Load the automobile data set as a pd.DataFrame."""
  # Download and cache the data
  path = URL

  # Load it into a pandas DataFrame
  df = pd.read_csv(path, names=COLUMN_TYPES.keys(),
                   dtype=COLUMN_TYPES, na_values="?", skiprows=1)

  return df


def load_data(y_name="Label", train_fraction=0.7, seed=None):
  """Load the automobile data set and split it train/test and features/label.

  A description of the data is available at:
    https://archive.ics.uci.edu/ml/datasets/automobile

  The data itself can be found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the data set to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = load_data(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features DataFrame.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)


def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a slice Dataset from a pandas DataFrame and labels"""

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn