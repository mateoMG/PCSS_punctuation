from transformers import BertTokenizer
from bs4 import BeautifulSoup
from tqdm import tqdm
import string
import glob
import gzip
import os
import shutil

"""""
Define global variable - set() with words start with capital letter 
"""""
upper_words = set()
with gzip.open('polimorf-20220403.tab.gz', 'rt', encoding='utf8') as dict:
    for lines in dict:
        if lines.strip():
            #print(line.split()[len(line.split()) - 1]) #Gatunek słowa
            if lines[0].isupper() and lines.split()[len(lines.split()) - 1] in {"nazwa_geograficzna"}:
                upper_words.add(lines.split()[0].lower())


def untokenization(words, punctuation, name):
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
    special_sign = False
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
        elif t in {'>', ';', ')'}:
            new_text += t
            big_letter = False
        elif t in {'<', '('}:
            new_text += " " + t
            big_letter = False
            special_sign = True
        elif t in {'-', '_', ':'}:
            new_text += t
            big_letter = False
            special_sign = True
        else:
            if special_sign:
                new_text += t
                big_letter = False
                special_sign = False
            else:
                new_text += " " + t
                big_letter = False
    #new_text = postprocessing(new_text, upper_words)
    "Save text"
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
                
def prepare_xml_output_folder(original, destination):
    """""
    Create folder for output xml files
    """""
    isDir = os.path.isdir(destination)
    if isDir:
        shutil.rmtree(destination)
    shutil.copytree(original, destination)
    
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

def xml_writer(test_path, xml_result_path):
    """""
    Convert txt to xml
    """""
    for file in glob.glob(test_path + "/*txt"):
        for filex in glob.glob(xml_result_path + "/*.xml"):
            if os.path.basename(file)[:-4] == os.path.basename(filex)[:-4]:
                with open(file, 'r', encoding='utf8') as f:
                    fi = f.readlines()
                    fi = " ".join(fi).split()
                    with open(filex, 'r+', encoding='utf8') as x:
                        file_xml = x.read()
                        data = BeautifulSoup(file_xml, 'xml')
                        xml_word = data.find_all('Word')
                        i = 0
                        for d in xml_word:
                            words = d.text.split()
                            d.string = " ".join(fi[i:i+len(words)])
                            i += len(words)
                    with open(filex, "w", encoding='utf-8') as fil:
                        fil.write(str(data))
                        break

def postprocessing(text, upper_words):
    """""
    Add capital letters for names, countries etc.
    """""
    print("Capital letters:")
    text = text.split()
    counter = 0
    punctuation_mark = False
    for word_index in tqdm(range(len(text))):
        if text[word_index][len(text[word_index])-1] in {".", ",", "?"}:
            save_punc = text[word_index][len(text[word_index])-1]
            punctuation_mark = True
            text[word_index]= text[word_index][:len(text[word_index])-1]
        if text[word_index] in upper_words:
            text[word_index] = string.capwords(text[word_index])
            counter += 1
        if punctuation_mark:
            text[word_index] += save_punc
            punctuation_mark = False
    text = ' '.join(text)
    print("Add {} big letters in text".format(counter))
    return text
