from transformers import BertTokenizer
from bs4 import BeautifulSoup
import string
import glob
import os

def untokenization(words, punctuation,name):
    """""
    Create text from tokens
    """""
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    tokens = []
    index = 0
    for word in words:
        tokens.append(tokenizer.convert_ids_to_tokens(word))
        if punctuation[index] != 0:
            if punctuation[index] == 1:
                tokens.append(tokenizer.convert_ids_to_tokens(16)) #1010
            elif punctuation[index] == 2:
                tokens.append(tokenizer.convert_ids_to_tokens(18)) #1012
            elif punctuation[index] == 3:
                tokens.append(tokenizer.convert_ids_to_tokens(35)) #1029
        index += 1
    new_text = ""
    big_letter = True
    for t in tokens:
        t = t.lower()
        if big_letter:
            t = string.capwords(t)
        if t.startswith('##'):
            new_text += t[2:]
            big_letter = False
        elif t == "." or t == "?":
            new_text += t
            big_letter = True
        elif t == ",":
            new_text += t
            big_letter = False
        else:
            new_text += " " + t
            big_letter = False
    with open(name, 'w') as f:
        f.write(new_text)

def cut_lines(text_name, new_text_name, number_of_lines):
    """""
    Cut lines
    """""
    with open(text_name, 'r', encoding='utf8') as f:
        with open(new_text_name, 'w', encoding='utf8') as k:
            for line in f:
                if number_of_lines <= 0:
                    break
                k.write(line)
                number_of_lines -= 1

def xml_reader(test_path):
    """""
    Convert xml to txt
    """""
    for file in glob.glob(test_path + "/*.xml"):
        new_file = "inter_tset/dotestu/" + os.path.basename(file)[:-4] + ".txt"
        k = open(new_file, "w")
        with open(file, 'r') as f:
            file_xml = f.read()
        data = BeautifulSoup(file_xml, 'xml')
        data = data.find_all('Word')
        for d in data:
            print(d.text, file=k, end=" ")






