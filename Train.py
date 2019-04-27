import iris_data
import pandas as pd
import tensorflow as tf

train_path, test_path = iris_data.maybe_download()
print(train_path)

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[1:]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns, n_classes=2)

# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : iris_data.csv_input_fn(train_path, batch_size))