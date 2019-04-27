"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')


def main(argv):
  """Builds, trains, and evaluates the model."""
  args = parser.parse_args(argv[1:])

  (train_x,train_y), (test_x, test_y) = data.load_data()


  # Provide the training input dataset.
  train_input_fn = data.make_dataset(args.batch_size, train_x, train_y, True, 1000)

  # Provide the validation input dataset.
  test_input_fn = data.make_dataset(args.batch_size, test_x, test_y)

  # Use the same categorical columns as in `linear_regression_categorical`
  body_style_vocab = ["47980dda", "79ceee49"]
  body_style_column = tf.feature_column.categorical_column_with_vocabulary_list(key="auction_boolean_0", vocabulary_list=body_style_vocab)
  #make_column = tf.feature_column.categorical_column_with_hash_bucket(
      #key="make", hash_bucket_size=50)

  feature_columns = [
      tf.feature_column.numeric_column(key="creative_height"),
      tf.feature_column.numeric_column(key="creative_width"),
      # Since this is a DNN model, categorical columns must be converted from
      # sparse to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      tf.feature_column.indicator_column(body_style_column)
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      #tf.feature_column.embedding_column(make_column, dimension=3),
  ]

  # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
  # defined above as input.
  model = tf.estimator.LinearClassifier(
    feature_columns=feature_columns, n_classes=2)

  # Train the model.
  # By default, the Estimators log output every 100 steps.
  model.train(input_fn=train_input_fn, steps=args.train_steps)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=test_input_fn)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}".format(average_loss))

  print()


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)