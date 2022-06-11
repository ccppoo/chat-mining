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

    대조군으로 사용하기 위해서 토크나이저들이 명사만 추출한 "nouns" 열,
    hanspell 라이브러리를 이용한 것 또한 있다.

※   1회의 스트리밍에 평균적으로 2~3만 개의 채팅 텍스트, 많게는 5만 적게는 1~2만 개로 짧지만 많은 텍스트를 다루다보니
    시간 효율성을 위해서 Multiprocess module을 사용하고 있다.
    이 코드를 실행하는 환경에 따라 Multiprocess module 사용 여부를 바꾸면 된다.
'''

from utils import *
from konlpy.tag import Okt, Mecab
import pandas as pd
from multiprocessing import Pool

HEADERS = ['real time','uptime', 'upsecond','nickname', 'chat', 'preprocessed', 'hanspell']
HEADERS = ['real time','uptime', 'upsecond','nickname', 'chat', 'preprocessed',]

MERGED = "MERGED_{}.txt"

def tokenizing(streamer : str, pool : Pool):
    '''

    '''
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    okt = Okt()

    NEW_HEADERS = list(HEADERS)

    NEW_HEADERS.append(MORPHS_MECAB)
    NEW_HEADERS.append(MORPHS_MECAB_NOUN)
    # NEW_HEADERS.append(HANSPELL_MECAB_NOUN)

    NEW_HEADERS.append(MORPHS_OKT)
    NEW_HEADERS.append(MORPHS_OKT_NOUN)
    # NEW_HEADERS.append(HANSPELL_OKT_NOUN)

    targets = [
        MORPHS_MECAB,MORPHS_MECAB_NOUN, HANSPELL_MECAB_NOUN, 
        MORPHS_OKT,MORPHS_OKT_NOUN, HANSPELL_OKT_NOUN
    ]

    targets = [
        MORPHS_MECAB,MORPHS_MECAB_NOUN,  
        MORPHS_OKT,MORPHS_OKT_NOUN
    ]

    files_to_do = list(getFilesFrom(FILE_PATH(PREPROCESSED, streamer, 'csv'), extension='csv'))

    def workTodo(i, dirpath, csvfile):
        works = f'{str(i).rjust(3)}/{len(files_to_do)}'.rjust(10)
        logging.info(f"streamer : {streamer.rjust(20)} | working {works} ({i/len(files_to_do)*100:.2f} %)")

        csvfile_path = pathlib.Path(dirpath, csvfile)
        df = pd.read_csv(csvfile_path)

        if all([x in df.columns for x in targets]):
            logging.info(f"passing csv file {streamer}::{csvfile} since already done")
            return

        df[MORPHS_MECAB] = df[PREPROCESSED].apply(lambda value: mecab.morphs(value.strip('""')))
        df[MORPHS_MECAB_NOUN] = df[PREPROCESSED].apply(lambda value: mecab.nouns(value.strip('""')))
        # df[HANSPELL_MECAB_NOUN] = df[HANSPELL].apply(lambda value: mecab.nouns(value.strip('""')))

        df[MORPHS_OKT] = df[PREPROCESSED].apply(lambda value: okt.morphs(value.strip('""')))
        df[MORPHS_OKT_NOUN] = df[PREPROCESSED].apply(lambda value: okt.nouns(value.strip('""')))
        # df[HANSPELL_OKT_NOUN] = df[HANSPELL].apply(lambda value: mecab.nouns(value.strip('""')))

        df = df[NEW_HEADERS]

        with open(csvfile_path, mode='w', newline='', encoding='utf-8') as fp:
            df.to_csv(fp, index=False)

    # for i, (dirpath, csvfile) in enumerate(files_to_do, start=1):
    #     workTodo(i, dirpath, csvfile)

    a = [(i, dirpath, csvfile) for i, (dirpath, csvfile) in enumerate(files_to_do, start=1)]
    pool.starmap(workTodo, a)
    logging.info(f'finished tokenizing streamer : {streamer}')

if __name__ == '__main__':
    '''
    멑티 프로세싱을 동시에 여러스트리머를 대상으로 하는 것이 아니라
    한번에 스트리머의 채팅 데이터 여러개를 대상으로 한다.

    스트리머마다 가지고 있는 파일의 크기가 비슷하다면 문제가 없지만,
    스트리머마다 채팅 데이터의 크기가 상이해서 스트리머 기준으로 할 경우
    제일 큰 스트리머들의 작업만 남음

    예) 
        작업중 : 침착맨, 플러리, 김도 (X) 
        작업중 : 침착맨 (2341.csv , 4412.csv, 9491.csv) (O)
    '''

    # from multiprocessing import Pool

    sts = [x for x in getStreamers()]
    
    pool = Pool(4)
    # pool.map(tokenizing, sts)

    for streamer in sts:
        tokenizing(streamer, pool)