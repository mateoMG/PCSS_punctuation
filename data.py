from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm

def load(filename):
    """""
    Load text file
    """""
    print("Load file: {}".format(filename))
    f = open(filename, "r", encoding='utf-8')
    data = f.readlines()
    number_of_lines(data)
    return data

def number_of_lines(file):
    """""
    Count number of lines
    """""
    counter = 0
    for line in file:
        counter += 1
    print("Number of lines in the file: {}".format(counter))

def find_punctuation(punctuations, tokenizer):
    """""
    This function allows to find ids of punctuation symbols tokens
    """""
    punc = []
    for name in punctuations.keys():
        punc.append(tokenizer.convert_tokens_to_ids(name))
    return punc

def end_of_line_token(data, tag, tokenizer):
    """""
    Is it necessary or helpful ??
    """""
    end_token = tokenizer.convert_tokens_to_ids('[SEP]')
    data.append(end_token)
    tag.append(0)
    return data, tag

def tagging(x, p_numbers):
    """""
    Tag every word in the sentence 
    """""
    data = []
    tag = []
    for i in range(len(x)-1):
        var_1 = False
        for k in range(len(p_numbers)):
            if x[i+1] == p_numbers[k]:
                tag.append(k+1)
                var_1 = True
                break
        if not var_1:
            tag.append(0)
        data.append(x[i])
    data.append(x[len(x)-1])
    tag.append(0)
    delete_punc(data, tag, p_numbers)
    return data, tag

def delete_punc(data, tag, p_numbers):
    """""
    Delete punctuation's ids and tags
    """""
    index = []
    for i in range(len(data)):
        for k in range(len(p_numbers)):
            if data[i] == p_numbers[k]:
                index.append(i)
                break
    for i in range(len(index)):
        data.pop(index[i]-i)
        tag.pop(index[i]-i)

def split_dataset(data, tag, p):
    """""
    Create train and test datasets
    """""
    data_test = data[(len(data)-len(data)//p):]
    del data[(len(data)-len(data)//p):]
    tag_test = tag[(len(tag)-len(tag)//p):]
    del tag[(len(tag)-len(tag)//p):]
    return data, tag, data_test, tag_test

def create_data(path):
    """""
    General function to create lists with data (words) and tags
    """""
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    text = load(path)
    punctuation = {',': 1,
                   '.': 2,
                   '?': 3}
    X = []
    Y = []
    punctuation_ids = find_punctuation(punctuation, tokenizer)
    for line in tqdm(text):
        word = line
        word = word.lower() #for target texts without capital letters
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        #print(word)
        #print(x)
        #print(tokenizer.convert_ids_to_tokens(x))
        x, y = tagging(x, punctuation_ids)
        #x, y = end_of_line_token(x, y, tokenizer) #add SEC token
        X.extend(x)
        Y.extend(y)
    print("Number of words in dataset: {}".format(len(X)))
    print("Number of tags: {}".format(len(Y)))
    return X, Y

def to_one_hot(tags, dimension):
    """""
    One hot encoding
    """""
    results = np.zeros((len(tags), dimension))
    for idx, tag in enumerate(tags):
        results[idx, tag] = 1
    return results

def batch_creator(data, tags, m, batch_size, num_classes):
    """""
    Batch creator: (i - (m - 1) // 2; i + (m - 1) // 2) sequences + labels
    """""
    sample_idx = 0
    while sample_idx < len(data):
        if sample_idx > (len(data)-batch_size):
            samples = len(data) - sample_idx
        else:
            samples = batch_size
        inputs = np.empty((samples, m), dtype=np.int64)
        inputs.fill(0)
        for batch_idx in range(samples):
            list_of_tokens = []
            for i in range(m):
                i = i + batch_idx
                if i - (m - 1) // 2 < 0:
                    list_of_tokens.append(0)
                elif i - (m - 1) // 2 > len(data) - 1:
                    list_of_tokens.append(0)
                else:
                    list_of_tokens.append(data[i - (m - 1) // 2])
            inputs[batch_idx, :] = list_of_tokens
            sample_idx += batch_size
    tags = to_one_hot(tags, num_classes)
    return inputs, tags
















