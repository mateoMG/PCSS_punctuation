import glob
import tensorflow as tf
import torch
from transformers import TFBertForSequenceClassification
from sklearn.metrics import recall_score, precision_score, f1_score

from data import create_data, split_dataset, batch_creator
from tools import prepare_xml_output_folder, untokenization, xml_reader, xml_writer

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
    "Convert xml to txt"
    test_xml_path = 'inter_tset/xmltest'
    destination_xml_path = 'inter_tset/wyniki_xml'
    xml_reader(test_xml_path)
    prepare_xml_output_folder(test_xml_path, destination_xml_path)
    

    "Load test path"
    test_path = "inter_tset/dotestu"
    subsequence_length = 5
    num_classes = 4

    "Train model"
    model = model(num_classes)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.CategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.load_weights('bert_weights_2.h5')

    for file in glob.glob(test_path + "/*.txt"):

        X_test, Y_test = create_data(file)
        print("Test dataset X: {} Test labels Y: {}".format(len(X_test), len(Y_test)))
        inputs_test, labels_test = batch_creator(X_test, Y_test, subsequence_length, batch_size=len(X_test), num_classes=num_classes)
        print("Test input X: {} Test labels Y: {}".format(inputs_test.shape, labels_test.shape))

        evaluation = model.evaluate(inputs_test, labels_test)
        predictions = model.predict(inputs_test) #predictions for test dataset
        predictions, labels_test = stats(labels_test, predictions)

        "Untokenization"
        folder = "inter_tset/wyniki/" + file[18:]
        untokenization(X_test, predictions, folder)

    xml_writer("inter_tset/wyniki", destination_xml_path)