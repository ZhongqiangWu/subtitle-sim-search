# coding: utf-8
"""根据查询词搜索语义相似的字幕"""

import sys
sys.path.append('../conf')
from argparse import ArgumentParser

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from conf import EMBEDDING_PATH, FAISS_INDEX_PATH, INDEX_DICT_PATH, TOPN
from utils import load_pickle


def output_one(idx, idx_dict, window_size=5):
    """输出一个结果及其上下文
    
    Args:
        idx: 整型，要输出的索引下表
        idx_dict: 字典，{idx: ChunkItem}
        window_size: 整型，输出上下文的窗口，以当前的台词为中心窗口的大小
    """
    right_idx = idx + int(window_size/2) + 1
    left_idx = idx - int(window_size/2)
    # 当前所属电影
    cur_movie = idx_dict[idx].movie
    # 处理边界情况
    if right_idx > len(idx_dict):
        left_idx = max((left_idx - (right_idx - len(idx_dict) + 1)), 0)
        right_idx = len(idx_dict) - 1
    if left_idx < 0:
        right_idx = min(right_idx-left_idx, len(idx_dict)-1)
        left_idx = 0
    for out_idx in range(left_idx, right_idx):
        if cur_movie != idx_dict[out_idx].movie:
            continue
        print(' | '.join([idx_dict[out_idx].movie,
                          idx_dict[out_idx].timerange,
                          idx_dict[out_idx].subtitle]))


def main(query):
    model = BGEM3FlagModel(EMBEDDING_PATH,
                       use_fp16=True)
    faiss_idx = faiss.read_index(FAISS_INDEX_PATH)
    idx_dict = load_pickle(INDEX_DICT_PATH)
    print(len(idx_dict))
    vector = model.encode(query)['dense_vecs']
    vector = np.array([vector]).astype("float32")
    print(f'vector=', vector, len(vector))
    dist_list, idx_list = faiss_idx.search(vector, k=TOPN)
    print(f'query={query}')
    for idx, (_, recall_idx) in enumerate(zip(dist_list[0], idx_list[0])):
        print(f'============chunk_{idx+1}===============')
        output_one(recall_idx, idx_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-q', '--query', help="查询query", required=True)
    args = parser.parse_args()
    main(args.query)
