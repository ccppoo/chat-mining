# -*- coding: utf-8 -*-
# 2_토큰화.py

'''
이번 단계에서는 문장을 분해해 클러스터링을 하기위해 채팅을 벡터로 나타낼 수 있게 만들어주는 vocab을 만들기 전에
기존의 어휘 사전을 기반으로 토크나이징을 할 것이다.

이렇게 하는 이유는 기존의 문장을 구조적을로 잘 분해했었던 Okt, Mecab 라이브러리들이 채팅이라는 특이한 형태의 문장을 잘 파악할 수 있는지.
또 어휘 등장 빈도 기반으로 만들어진 Vocab이 더 성능이 좋을지 파악하기 위해서다.

(1) tokenizing

    위에 서술한 1~5번 과정을 하나로 종합한 함수다.
    이외 추가로 동일한 자료에 대하여 반복적인 토크나이징 작업을 피하기 위해서
    konlpy.tag 라이브러리의 Okt 토크나이저를 이용해 토큰화 작업을 시행해 "morphs" column에 추가했다.

※   1회의 스트리밍에 평균적으로 2~3만 개의 채팅 텍스트, 많게는 5만 적게는 1~2만 개로 짧지만 많은 텍스트를 다루다보니
    시간 효율성을 위해서 Multiprocess module을 사용하고 있다.
    이 코드를 실행하는 환경에 따라 Multiprocess module 사용 여부를 바꾸면 된다.
'''

import os
import pathlib
from functools import partial
from konlpy.tag import Okt, Mecab
import pandas as pd
import logging
logger = logging.getLogger()

HEADERS = ['real time','uptime', 'upsecond','nickname', 'chat', 'preprocessed']

CWD = os.getcwd()
ORIGIN_FILE_PATH = partial(pathlib.Path, CWD, 'data')
FILE_PATH = partial(pathlib.Path, CWD)
PREPROCESSED ='preprocessed'
MERGED = "MERGED_{}.txt"
UTF_8 = 'utf-8'

def getStreamers():
    for streamer in os.listdir(ORIGIN_FILE_PATH()):
        if os.path.isdir(ORIGIN_FILE_PATH(streamer)):
            yield streamer

def getFilesFrom(*paths, extension : str):
    dir_name = FILE_PATH(*paths)
    
    for filename in os.listdir(dir_name):
        if filename.endswith(extension):
            yield dir_name, filename

def tokenizing(streamer : str):
    '''

    '''
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    okt = Okt()

    MORPHS_MECAB = 'morphs_Mecab'
    MORPHS_OKT = 'morphs_Okt'

    NEW_HEADERS = list(HEADERS)
    NEW_HEADERS.append(MORPHS_MECAB)
    NEW_HEADERS.append(MORPHS_OKT)

    files_to_do = list(getFilesFrom(FILE_PATH(PREPROCESSED, streamer, 'csv'), extension='csv'))

    for i, (dirpath, csvfile) in enumerate(files_to_do, start=1):
        works = f'{str(i).rjust(3)}/{len(files_to_do)}'.rjust(10)
        logger.info(f"streamer : {streamer.rjust(20)} | working {works} ({i/len(files_to_do):.2f} %)")

        csvfile_path = pathlib.Path(dirpath, csvfile)
        df = pd.read_csv(csvfile_path)

        df[MORPHS_MECAB] = df['preprocessed'].apply(lambda value: mecab.morphs(value.strip('""')))
        df[MORPHS_OKT] = df['preprocessed'].apply(lambda value: okt.morphs(value.strip('""')))
    
        df = df[NEW_HEADERS]

        with open(csvfile_path, mode='w', newline='', encoding='utf-8') as fp:
            df.to_csv(fp, index=False)
        
        logger.info(f'finished tokenizing streamer : {streamer}')

if __name__ == '__main__':
    from multiprocessing import Pool

    sts = [x for x in getStreamers()]
    
    pool = Pool(len(sts))
    pool.map(tokenizing, sts)

    # for streamer in sts:
    #     tokenizing(streamer)