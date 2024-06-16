# coding: utf-8
"""将srt字幕向量化"""
import os
import sys
sys.path.append('../conf')

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from conf import SUBTITLE_DIR, EMBEDDING_PATH, FAISS_INDEX_PATH, INDEX_DICT_PATH
from utils import ChunkItem, write_pickle


def load_srt_single(file_dir, filename, idx_dict, subtitle_list):
    """加载单个srt字幕文件
    
    """

    file_path = os.path.join(file_dir, filename)
    print(f'[INFO] load_srt_single() file_path={file_path}')

    # 标识是否字幕开始
    is_start = False
    # 存储字幕块
    subtitle_cell_list = []
    timerange, subtile_chunk = '', ''
    with open(file_path, 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            if '->' in line:
                is_start = True
                timerange = line.strip()
                continue
            # 字幕块结束，重置
            if line.strip() == '':
                is_start = False
                subtile_chunk = '\n'.join(subtitle_cell_list)
                subtitle_idx = len(subtitle_list)
                subtitle_list.append(subtile_chunk)
                idx_dict[subtitle_idx] = ChunkItem(timerange =timerange,
                                                   subtile=subtile_chunk,
                                                   movie=filename.replace('.srt', ''))
                subtitle_cell_list = []
                continue
            if is_start:
                subtitle_cell_list.append(line.strip())


def load_srt(data_dir):
    """加载srt字幕"""

    idx_dict = {}
    # 全量字幕列表
    subtitle_list = []
    for sub_dir in os.listdir(data_dir):
        if sub_dir.startswith('.'):
            continue
        sub_path = os.path.join(data_dir, sub_dir)
        # 如果是文件夹，逐一遍历
        if os.path.isdir(sub_path):
            for filename in os.listdir(sub_path):
                if filename.startswith('.'):
                    continue
                load_srt_single(file_dir=sub_path,
                                filename=filename,
                                idx_dict=idx_dict,
                                subtitle_list=subtitle_list)
        else:
            load_srt_single(file_dir=data_dir,
                            filename=sub_dir,
                            idx_dict=idx_dict,
                            subtitle_list=subtitle_list)
    print(f'subtitle_list={subtitle_list[:3]}')
    return idx_dict, subtitle_list


def text2vec(subtitle_list, model):
    """text -> vector"""
    vector_list = []
    print('len(subtitle_list)=', len(subtitle_list))
    for idx, subtitle in enumerate(subtitle_list):
        if idx % 10 == 0 :
            print(idx)
        vector = model.encode(subtitle)['dense_vecs']
        vector_list.append(vector)
    vector_list = np.array(vector_list)
    return vector_list


def main():
    faiss_index = faiss.IndexFlatL2(1024)
    faiss_index = faiss.IndexIDMap(faiss_index)
    idx_dict, subtitle_list = load_srt(SUBTITLE_DIR)
    print(len(idx_dict), len(subtitle_list))
    model = BGEM3FlagModel(EMBEDDING_PATH, use_fp16=True)
    subtitle_vec_list = text2vec(subtitle_list, model)
    ids = np.array(list(range(len(subtitle_vec_list))))
    faiss_index.add_with_ids(subtitle_vec_list, ids)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    write_pickle(idx_dict, INDEX_DICT_PATH)
    print(subtitle_vec_list)


if __name__ == '__main__':
    main()
