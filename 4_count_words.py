# -*- coding: utf-8 -*-
# 4_count_words.py

'''
서브워드로 이루어진 문장은 만들었으므로 문장을 벡터화하고 빈도 최상위 500개를 기준으로 벡터화하여
많이 등장하는 채팅의 형태의 클러스터를 잘 구축할 수 있는지 시도할 것이다.

Mecab, Okt를 이용한 토큰화된 결과물을 기준으로 파이썬 딕셔너리를 만든다.
나중에 최상위 빈도의 단어를 기준으로 벡터를 만들 때 사용되거나
최다빈도 단어, 희소 빈도 단어를 보기 위해서 사용된다.

하나의 스트리머에서 수집된 채팅을 기준으로한다. (방송 회차 구분 X)
'''

import pickle
from utils import *
import pandas as pd
import csv

DATA_FOLDER = FILE_PATH(PREPROCESSED)

def updateDict(dic : dict, line):
    for word in line:
        word = word.lower()
        if dic.get(word, None):
            dic[word] += 1
        else:
            dic[word] = 1

def saveDictAsPickle(streamer, this, tail_name):
    name = f'dict_{tail_name}.pickle'

    saveAt = FILE_PATH(PREPROCESSED, streamer, 'python_dict')
    makeDirIfNotExists(saveAt)

    saveto = FILE_PATH(PREPROCESSED, streamer, 'python_dict', name)
    with open(saveto, mode='wb') as fp:
        pickle.dump(this, fp)

def saveListAsPickle(streamer, this, tail_name):
    name = f'list_{tail_name}.pickle'

    saveAt = FILE_PATH(PREPROCESSED, streamer, 'python_list')
    makeDirIfNotExists(saveAt)
    
    saveto = FILE_PATH(PREPROCESSED, streamer,'python_list', name)
    with open(saveto, mode='wb') as fp:
        pickle.dump(this, fp)

def saveAsCSV(streamer, this : dict, name):
    
    makeDirIfNotExists(FILE_PATH(PREPROCESSED,streamer,'csv_for_visualize'))
    saveto = FILE_PATH(PREPROCESSED,streamer,'csv_for_visualize', name)

    tocsv = [['word', 'count']]
    [tocsv.append([k, v]) for k, v in this.items()]

    with open(saveto, mode='w', newline='', encoding=UTF_8) as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerows(tocsv)

def getTop(dict_ : dict, *tops : int):

    sortedDict = sorted(dict_.items(), key = lambda x : x[1], reverse=True)

    items = dict()

    for topRank in tops:
        items[topRank] = sortedDict[:topRank]
    
    return items

def doit(streamer):

    dct_Mecab = dict()
    dct_Mecab_noun = dict()
    dct_Okt = dict()
    dct_Okt_noun = dict()

    logging.info(f'Starting {streamer}')
    '''
    모든 파일을 읽고 하나로 통합하는 것이기 때문에 그냥 귀찮아서 동시성은 구현 안함
    '''

    length = len(list(getFilesFrom(PREPROCESSED, streamer, 'csv', extension='csv')))

    for i, (dirname, csvfile) in enumerate(getFilesFrom(PREPROCESSED, streamer, 'csv', extension='csv'), start=1):
        logging.info(f'Working on {streamer} {i}/{length} ({i/length*100:.2f}%)')

        df = pd.read_csv(FILE_PATH(dirname, csvfile))
        # morphs_Mecab, morphs_Okt
        for words in df[MORPHS_MECAB].to_list():
            wordList : List[str] =  eval(words)
            updateDict(dct_Mecab, wordList)

        for words in df[MORPHS_MECAB_NOUN].to_list():
            wordList : List[str] =  eval(words)
            updateDict(dct_Mecab_noun, wordList)
            
        for words in df[MORPHS_OKT].to_list():
            wordList : List[str] =  eval(words)
            updateDict(dct_Okt, wordList)

        for words in df[MORPHS_OKT_NOUN].to_list():
            wordList : List[str] =  eval(words)
            updateDict(dct_Okt_noun, wordList)

    logging.info(f'Saving {streamer} file..')

    for types in [dct_Mecab, dct_Mecab_noun,dct_Okt,dct_Okt_noun]:
        items = getTop(types, 500, 1000, 2000)
        saveListAsPickle(streamer, items[500], 'top_500')
        saveListAsPickle(streamer, items[1000], 'top_1000')
        saveListAsPickle(streamer, items[2000], 'top_2000')

    # 나중에 시각화하는 용도로 사용
    saveAsCSV(streamer,dct_Mecab, f'tokens_count_dct_Mecab.csv')
    saveAsCSV(streamer,dct_Mecab_noun, f'tokens_count_dct_Mecab_noun.csv')
    saveAsCSV(streamer,dct_Okt, f'tokens_count_dct_Okt.csv')
    saveAsCSV(streamer,dct_Okt_noun, f'tokens_count_dct_Okt_noun.csv')

    saveDictAsPickle(streamer, dct_Mecab, 'mecab')
    saveDictAsPickle(streamer, dct_Mecab_noun, 'mecab_noun')
    saveDictAsPickle(streamer, dct_Okt, 'okt')
    saveDictAsPickle(streamer, dct_Okt_noun, 'okt_noun')

    logging.info(f'*-- Finshed {streamer} --*')

if __name__ == '__main__':
    import multiprocessing as mp

    # for stmr in getStreamers():
    #     doit(stmr)
    mp.Pool(3).map(doit, [stmr for stmr in getStreamers()])
