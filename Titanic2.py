# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import argparse
import tensorflow as tf
import pandas as pd
import random

TRAIN_PATH = "Kaggle/ctr_data/ctr_20.csv"
EVALUATE_PATH = "Kaggle/ctr_data/ctr_21.csv"
PREDICT_PATH = "Kaggle/ctr_data/ctr_test.csv"
MODEL_PATH = "Checkpoints/Model2"

CSV_COLUMNS_TRAIN = ["Label","action_categorical_0","action_categorical_1","action_categorical_2","action_categorical_3","action_categorical_4","action_categorical_5","action_categorical_6","action_categorical_7","action_list_0","action_list_1","action_list_2","auction_age","auction_bidfloor","auction_boolean_0","auction_boolean_1","auction_boolean_2","auction_categorical_0","auction_categorical_1","auction_categorical_10","auction_categorical_11","auction_categorical_12","auction_categorical_2","auction_categorical_3","auction_categorical_4","auction_categorical_5","auction_categorical_6","auction_categorical_7","auction_categorical_8","auction_categorical_9","auction_list_0","auction_time","creative_categorical_0","creative_categorical_1","creative_categorical_10","creative_categorical_11","creative_categorical_12","creative_categorical_2","creative_categorical_3","creative_categorical_4","creative_categorical_5","creative_categorical_6","creative_categorical_7","creative_categorical_8","creative_categorical_9","creative_height","creative_width","device_id","device_id_type","gender","has_video","timezone_offset"]
CSV_COLUMNS_TEST = ["action_categorical_0","action_categorical_1","action_categorical_2","action_categorical_3","action_categorical_4","action_categorical_5","action_categorical_6","action_categorical_7","action_list_0","action_list_1","action_list_2","auction_age","auction_bidfloor","auction_boolean_0","auction_boolean_1","auction_boolean_2","auction_categorical_0","auction_categorical_1","auction_categorical_10","auction_categorical_11","auction_categorical_12","auction_categorical_2","auction_categorical_3","auction_categorical_4","auction_categorical_5","auction_categorical_6","auction_categorical_7","auction_categorical_8","auction_categorical_9","auction_list_0","auction_time","creative_categorical_0","creative_categorical_1","creative_categorical_10","creative_categorical_11","creative_categorical_12","creative_categorical_2","creative_categorical_3","creative_categorical_4","creative_categorical_5","creative_categorical_6","creative_categorical_7","creative_categorical_8","creative_categorical_9","creative_height","creative_width","device_id","device_id_type","gender","has_video","timezone_offset","id"]


def handle_data(data):
    #data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    #data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["auction_age"] = data["auction_age"].fillna(0)

    for key in data.keys():
        data[key] = data[key].fillna("nada")

    #data.pop("auction_age")
    data.pop("creative_height")
    data.pop("creative_width")
    data.pop("timezone_offset")


def load_data():
    train = pd.read_csv(TRAIN_PATH, header=0, skiprows=lambda i: i>0 and random.random() > 0.3, verbose= True)[CSV_COLUMNS_TRAIN]
    handle_data(train)
    train_x, train_y = train, train.pop('Label')

    evaluate = pd.read_csv(EVALUATE_PATH, header=0, skiprows=lambda i: i>0 and random.random() > 0.1, verbose= True)[CSV_COLUMNS_TRAIN]
    handle_data(evaluate)
    evaluate_x, evaluate_y = evaluate, evaluate.pop('Label')

    predict = pd.read_csv(PREDICT_PATH, header=0, skiprows=lambda i: i>0 and random.random() > 0.01, verbose= True)[CSV_COLUMNS_TEST]
    handle_data(predict)
    predict_x, num_id = predict, predict.pop('id')

    return (train_x, train_y), (evaluate_x, evaluate_y), (num_id, predict_x)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
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

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=30000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (evaluate_x, evaluate_y), (num_id, predict_x) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []

    #my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # First, convert the raw input to a numeric column.
    numeric_feature_column = tf.feature_column.numeric_column("auction_age")
    bucketized_feature_column = tf.feature_column.bucketized_column(
        source_column=numeric_feature_column,
        boundaries=[1, 11, 21, 31, 41, 51, 61, 99])
    my_feature_columns.append(bucketized_feature_column)



    for i in range(3):
        make_column = tf.feature_column.categorical_column_with_hash_bucket(key="auction_boolean_" + str(i), hash_bucket_size=3)
        my_feature_columns.append(tf.feature_column.embedding_column(make_column, dimension=3))


    for i in range(13):
        make_column = tf.feature_column.categorical_column_with_hash_bucket(key="auction_categorical_" + str(i),
                                                                            hash_bucket_size=10)
        my_feature_columns.append(tf.feature_column.embedding_column(make_column, dimension=10))
        make_column = tf.feature_column.categorical_column_with_hash_bucket(key="creative_categorical_" + str(i),
                                                                            hash_bucket_size=10)
        my_feature_columns.append(tf.feature_column.embedding_column(make_column, dimension=10))

    #for i in range(8):
        #make_column = tf.feature_column.categorical_column_with_hash_bucket(key="action_categorical_" + str(i), hash_bucket_size=10)
        #my_feature_columns.append(tf.feature_column.embedding_column(make_column, dimension=10))

    body_style_vocab = ["m", "f", "o", "nada"]
    body_style_column = tf.feature_column.categorical_column_with_vocabulary_list(key="gender",
                                                                                  vocabulary_list=body_style_vocab)
    my_feature_columns.append(tf.feature_column.indicator_column(body_style_column))


    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Three hidden layers of 10 nodes.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=2,
        model_dir= MODEL_PATH)

    # Train the Model.
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, args.batch_size),steps=args.train_steps)
    print('train complete')

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(evaluate_x, evaluate_y, args.batch_size))
    print('\nTest set accuracy: {accuracy:0.4f}\n'.format(**eval_result))

    predictions = classifier.predict(input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    prediction_ids = []
    probabilities = []
    probabilitiesOfYes = []

    for pred_dict in predictions:
        #class_id = pred_dict['class_ids'][0]
        #probability = pred_dict['probabilities'][class_id]
        probabilityOfYes = pred_dict['probabilities'][1]

        #prediction_ids.append(class_id)
        #probabilities.append(probability)
        probabilitiesOfYes.append(probabilityOfYes)

    #prediction_ids = [prediction['class_ids'][0] for prediction in predictions]
    #probabilities = [prediction['probabilities'][0] for prediction in predictions]

    submission = pd.DataFrame({
        "id": num_id,
        #"Label": prediction_ids,
        #"Probability": probabilities,
        "ProbabilityOfYes": probabilitiesOfYes,
    })
    submission.to_csv("nn_submission.csv", index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)