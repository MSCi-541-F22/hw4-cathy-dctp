### Your code for the IndexEngine file ###
import argparse
from collections import Counter, defaultdict
from genericpath import isdir, isfile
import gzip
from lib2to3.pytree import convert
import os, os.path
import re
import pickle
from document import Document


ALL_TAGS = {"<DOC>", "</DOC>", "<DOCNO>", "</DOCNO>", "<DOCID>", "</DOCID>", 
            "<HEADLINE>", "</HEADLINE>", "<TEXT>", "</TEXT>", "<GRAPHIC>", "</GRAPHIC>"}

# Lexicon 
TERM_ID = {}
ID_TERM = {}

INVERTED_IDX = defaultdict(list)


def verify_datapath(arg):
    if os.path.isfile(arg):
        return arg
    
    error_message = f"The data filepath [{arg}] does not exist."
    if os.path.isdir(arg):
        error_message = "The data filepath must be a file."
    
    raise argparse.ArgumentTypeError(error_message)


def verify_output_dir(arg):
    if os.path.isdir(arg):
        error_message = "The directory already exists."
        raise argparse.ArgumentTypeError(error_message)
    return arg


def is_closing_tag(tag):
    return True if tag[1] == '/' else False

# https://stackoverflow.com/questions/23793987/write-file-to-a-directory-that-doesnt-exist       
def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def process_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_filepath", type=verify_datapath, help="The original data file path")
    parser.add_argument("output_dir", type=verify_output_dir, help="The output directory for the documents")
    return parser.parse_args()


def process_input_datafile(input_filepath):
    with gzip.open(input_filepath, 'rt') as f:
        file_content = f.read()
    return file_content

# break doc into tokens
def tokenize(str):
    tokens = []
    str = str.lower()
    start = 0
    for cur in range(len(str)):
        if not str[cur].isalnum():
            if cur != start:
                tokens.append(str[start:cur])
            start = cur + 1
        if str[cur].isalnum() and cur == len(str) - 1:
            tokens.append(str[start:cur + 1])
    return tokens
            

# convert tokens to ids 
def count_tokens(tokens, word_counts):
    for token in tokens:
        if token not in TERM_ID:
            TERM_ID[token] = len(TERM_ID)
        word_counts[TERM_ID[token]] += 1
        

# Add the doc id and the number of occurrence to the postings 
def add_to_postings(word_counts, doc_id):
    for token_id, count in word_counts.items():
        INVERTED_IDX[token_id].append((doc_id, count))
    

def process_documents(file_content, output_dir):
    # return a list of Document objects and write documents into files
    id_doc = {}
    id_docno = {}
    docno_id = {}
    docno_wordcounts = {}
    # find list of tags
    pattern = r"<\/?[A-Z]+>"
    tags = re.finditer(pattern, file_content)
    
    tag_stack = []
    cur_doc = Document()
    word_counts = defaultdict(int)
    for tag in tags:
        if tag[0] not in ALL_TAGS:
            continue
        if not is_closing_tag(tag[0]):
            tag_stack.append(tag)
        else:
            open_tag = tag_stack.pop()
            if tag[0] == "</DOCNO>":
                cur_doc.docno = file_content[open_tag.end() + 1: tag.start() - 1]
                cur_doc.month = cur_doc.docno[2:4]
                cur_doc.date = cur_doc.docno[4:6]
                cur_doc.year = cur_doc.docno[6:8]
            if tag[0] == "</DOCID>":
                cur_doc.docid = file_content[open_tag.end() + 1: tag.start() - 1]
            if tag[0] == "</HEADLINE>":
                headline = re.sub(r"<\/?[P]>", "", file_content[open_tag.end() + 1: tag.start() - 1])
                cur_doc.headline = headline
                tokens = tokenize(headline)
                count_tokens(tokens, word_counts)
            if tag[0] == "</TEXT>":
                tokens = tokenize(re.sub(r"<\/?[A-Z]+>", "", file_content[open_tag.end() + 1: tag.start() - 1]))
                count_tokens(tokens ,word_counts)
            if tag[0] == "</GRAPHIC>":
                tokens = tokenize(re.sub(r"<\/?[A-Z]+>", "", file_content[open_tag.end() + 1: tag.start() - 1]))
                count_tokens(tokens, word_counts)
            if tag[0] == "</DOC>":
                # create raw file
                raw_doc = file_content[open_tag.start() : tag.end()]
                raw_file_path = f'{output_dir}/{cur_doc.year}/{cur_doc.month}/{cur_doc.date}/{cur_doc.docno}'
                cur_doc.rawFilePath = raw_file_path
                with safe_open_w(raw_file_path) as f:
                    f.write(raw_doc)
                
                # map docid with doc object 
                id_doc[cur_doc.docid] = cur_doc
                id_docno[cur_doc.docid] = cur_doc.docno
                docno_id[cur_doc.docno] = cur_doc.docid
                docno_wordcounts[cur_doc.docno] = sum(word_counts.values())
                
                # update inverted index 
                add_to_postings(word_counts, cur_doc.docid)
                
                word_counts = defaultdict(int)
                cur_doc = Document()
    
    # create a reverse map for TERM_ID
    # https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping            
    ID_TERM = {v: k for k, v in TERM_ID.items()}
    
    return [id_doc, id_docno, docno_id, docno_wordcounts, INVERTED_IDX, TERM_ID, ID_TERM]


def write_metadata_file(dicts, output_dir):
    # Get a list of document objects, write into a metadata file
    # https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    with open(f'{output_dir}/metadata.pkl', 'wb') as handle:
        pickle.dump(dicts, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    
def main():
    args = process_argument()
    file_content = process_input_datafile(args.data_filepath)
    dicts = process_documents(file_content, args.output_dir)
    write_metadata_file(dicts, args.output_dir)
        

if __name__ == "__main__":
    main()