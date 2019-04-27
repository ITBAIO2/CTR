import pandas as pd
import tensorflow as tf

TRAIN_URL = "/Users/agustinalbinati/Documents/developer/TensorFlow/Matias/Kaggle/ctr_data/test.csv"
TEST_URL = "/Users/agustinalbinati/Documents/developer/TensorFlow/Matias/Kaggle/ctr_data/test.csv"

CSV_COLUMN_NAMES = ["Label","action_categorical_0","action_categorical_1","action_categorical_2","action_categorical_3","action_categorical_4","action_categorical_5","action_categorical_6","action_categorical_7","action_list_0","action_list_1","action_list_2","auction_age","auction_bidfloor","auction_boolean_0","auction_boolean_1","auction_boolean_2","auction_categorical_0","auction_categorical_1","auction_categorical_10","auction_categorical_11","auction_categorical_12","auction_categorical_2","auction_categorical_3","auction_categorical_4","auction_categorical_5","auction_categorical_6","auction_categorical_7","auction_categorical_8","auction_categorical_9","auction_list_0","auction_time","creative_categorical_0","creative_categorical_1","creative_categorical_10","creative_categorical_11","creative_categorical_12","creative_categorical_2","creative_categorical_3","creative_categorical_4","creative_categorical_5","creative_categorical_6","creative_categorical_7","creative_categorical_8","creative_categorical_9","creative_height","creative_width","device_id","device_id_type","gender","has_video","timezone_offset"]
SPECIES = ['No Compra',"Compra"]

def maybe_download():
    #train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    #test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    train_path = TRAIN_URL
    test_path = TEST_URL

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0], [0.0], [0.0], [0.0], [0.0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES, select_cols=[0,1,2,3,4])

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES[:5], fields))

    # Separate the label from the features
    label = features.pop('Label')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset