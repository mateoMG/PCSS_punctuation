import tensorflow as tf
import torch
from transformers import TFBertForSequenceClassification
from sklearn.metrics import recall_score, precision_score, f1_score

from data import create_data, split_dataset, batch_creator
from tools import cut_lines, untokenization

def model(num_labels):
    #model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    #model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
    model = TFBertForSequenceClassification.from_pretrained("dkleczek/bert-base-polish-cased-v1", num_labels=num_labels, from_pt=True)
    model.summary()
    return model

def stats(labels_test, predictions):
    predictions = tf.nn.softmax(predictions).numpy()
    predictions = predictions[0].argmax(axis=1)
    labels_test = labels_test.argmax(axis=1)

    cm = tf.math.confusion_matrix(labels_test, predictions)
    print("Confusion matrix: ")
    print(cm)
    print("Recall: ")
    print(recall_score(labels_test, predictions, average=None)) #macro to find a mean
    print("Precision: ")
    print(precision_score(labels_test, predictions, average=None))
    print("F1 score: ")
    print(f1_score(labels_test, predictions, average=None))
    return predictions, labels_test

if __name__ == '__main__':

    "Load file and create train, val and test dataset"
    cut_lines("train.txt", "filefile.txt", 100000)
    path = "filefile.txt"
    #test_path = "inter_tset/pozcast/POZcast_13_punc.txt"
    test_path = "inter_tset/dotestu/Ukraina.txt"
    X_train, Y_train = create_data(path)
    X_test, Y_test = create_data(test_path)
    X_test, Y_test, X_val, Y_val = split_dataset(X_test, Y_test, p=2)
    #X_train, Y_train, X_test, Y_test = split_dataset(X_train, Y_train, p=10)
    print("Train dataset X: {} Train labels Y: {}".format(len(X_train), len(Y_train)))
    print("Val dataset X: {} Val labels Y: {}".format(len(X_val), len(Y_val)))
    print("Test dataset X: {} Test labels Y: {}".format(len(X_test), len(Y_test)))

    "Load inputs for model"
    subsequence_length = 5
    num_classes = 4
    inputs, labels = batch_creator(X_train, Y_train, subsequence_length, batch_size=len(X_train), num_classes=num_classes)
    inputs_val, labels_val = batch_creator(X_val, Y_val, subsequence_length, batch_size=len(X_val), num_classes=num_classes)
    inputs_test, labels_test = batch_creator(X_test, Y_test, subsequence_length, batch_size=len(X_test), num_classes=num_classes)
    print("Train input X: {} Train labels Y: {}".format(inputs.shape, labels.shape))
    print("Val input X: {} Val labels Y: {}".format(inputs_val.shape, labels_val.shape))
    print("Test input X: {} Test labels Y: {}".format(inputs_test.shape, labels_test.shape))

    "Train model"
    model = model(num_classes)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.CategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    history = model.fit(inputs, labels, batch_size=64, epochs=3, validation_data=(inputs_val, labels_val))
    model.save_weights('bert_weights_2.h5')

    "Evaluate model"
    evaluation = model.evaluate(inputs_test, labels_test)
    print(evaluation)  #loss and accuracy for test dataset
    predictions = model.predict(inputs_test) #predictions for test dataset
    predictions, labels_test = stats(labels_test, predictions)

    "Untokenization"
    untokenization(X_test, predictions, "new.txt")