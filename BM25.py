import argparse
import heapq
import math
import os
import pickle
import logging
from nltk.stem import *


FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def process_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_directory", help="The directory location of your index")
    parser.add_argument("queries", help="The queries file")
    parser.add_argument("output_file", help="The name of the file that stores the output")
    return parser.parse_args()

args  = process_arguments()
with open(f'{args.index_directory}/metadata.pkl', 'rb') as handle:
    doc_map = pickle.load(handle)
K1 = 1.2
B = 0.75
K2 = 7
DOCNO_ID = doc_map[2]
DOCNO_WORDCOUNTS = doc_map[3]   
INVERTED_IDX = doc_map[4]
TERM_ID = doc_map[5]

def get_inverted_idx_v2(inverted_idx):
    res = dict()
    for token_id in inverted_idx:
        for doc_id, freq in inverted_idx[token_id]:
            res[(token_id, doc_id)] =  freq
    return res

INVERTED_IDX_V2 = get_inverted_idx_v2(INVERTED_IDX)


# break query into tokens
def tokenize(str):
    stemmer = PorterStemmer()
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
    return [stemmer.stem(token) for token in tokens]
    # return tokens

    
def convert_to_token_ids(tokens):
    token_ids = []
    for token in tokens:
        # if any token in the query is not found in lexicon, 
        # no document containing all tokens in query 
        if token not in TERM_ID:
            continue
        token_ids.append(TERM_ID[token])
    return token_ids


def count_token(token_ids):
    token_count = {}
    for token_id in token_ids:
        if token_id not in token_count:
            token_count[token_id] = 1
        else:
            token_count[token_id] += 1
    return token_count
        

def calculate_k_map():
    # print(f"doc length: {DOCNO_WORDCOUNTS[docno]}")
    # print(f"average doc length: {(sum(DOCNO_WORDCOUNTS.values()) / len(DOCNO_WORDCOUNTS))}")
    k_map = dict()
    avg_wordcounts = sum(DOCNO_WORDCOUNTS.values()) / len(DOCNO_WORDCOUNTS)
    for docno in DOCNO_WORDCOUNTS:
        k_map[docno] = K1 * ((1 - B) + B * DOCNO_WORDCOUNTS[docno] / avg_wordcounts)
    return k_map
K_MAP = calculate_k_map()


def get_fi(token_id, docid):
    return INVERTED_IDX_V2.get((token_id, docid), 0)


def get_ni(token_id):
    return len(INVERTED_IDX[token_id])


def calculate_bm25(docno, token_ids, token_count):
    K = K_MAP[docno]
    # print(f"K: {K}")
    bm25 = 0 
    for token_id in token_ids:
        fi = INVERTED_IDX_V2.get((token_id, DOCNO_ID[docno]), 0) 
        ni = get_ni(token_id)
        qfi = token_count[token_id]
        # print(f"fi: {fi}")
        # print(f"ni: {ni}")
        # print(f"qfi: {qfi}")
        # print(f"number of docs: {len(DOCNO_WORDCOUNTS)}")
        bm25 += (
            ((K1 + 1) * fi / (K + fi)) * 
            ((K2 + 1) * qfi / (K2 + qfi)) * 
            math.log((len(DOCNO_WORDCOUNTS) - ni + 0.5) / (ni + 0.5))
        )
    return bm25     


def search_and_rank(query):
    # print(f"list of tokens: {tokenize(query)}")
    token_ids = convert_to_token_ids(tokenize(query))
    token_count = count_token(token_ids)
    results = []
    for docno in DOCNO_WORDCOUNTS:
        bm25 = calculate_bm25(docno, token_ids, token_count)
        heapq.heappush(results, (bm25, docno))
        if len(results) > 1000:
            heapq.heappop(results)
    return sorted(results, reverse=True)
    

def print_top_1000(topic_id, results, run_tag):
    output = []
    for i in range(len(results)):
        output.append(f"{topic_id} Q0 {results[i][1]} {i+1} {results[i][0]} {run_tag}")
    return output
        

def main():
    run_tag = "dctphamAND"
    logger.info(f"Start splitting queries")
    with open(f'{args.queries}') as f:
        lines = [line.strip() for line in f.readlines()] 
    logger.info("Finish splitting queries") 
          
    for i in range(0, len(lines), 2):
        topic_id = lines[i]
        query = lines[i + 1]
        logger.info(f"Start query {topic_id}")
        results = search_and_rank(query)
        output = print_top_1000(topic_id, results, run_tag)
        logger.info(f"Finish query {topic_id}. Writing to {args.output_file}.")
        with open(f'{args.output_file}', 'a') as f:
            f.write("\n".join(output))
            f.write("\n")

    
if __name__ == '__main__':
    main()