# coding: utf-8
"""辅助函数"""
import pickle


class ChunkItem:
    """字幕块类"""

    def __init__(self, timerange, subtile, movie) -> None:
        """
        Args:
            timerange: 字符串，时间范围
            subtile: 字符串，字幕文本
            movie: 字符串，所属电影名称
        """
        self.timerange = timerange
        self.subtitle = subtile
        self.movie = movie


def write_pickle(data, save_path):
    """保存将对象"""

    with open(save_path, 'wb') as fout:
        pickle.dump(data, fout)


def load_pickle(data_path):
    """获取对象"""

    with open(data_path, 'rb') as fin:
        data = pickle.load(fin)
    return data
