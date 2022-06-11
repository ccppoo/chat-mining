# -*- coding: utf-8 -*-
# 1_sentencepiece.py

'''
데이터를 접할 때 전제로 스트리밍 중 발생하는 단어는 사전에 등재되어 있지 않는 특성이 있다.
OOV(Out of Vocabulary) 문제를 해결하기 위해서 Word2Vec 또는 ELMo를 사용하는데
이와 유사한 대안으로 나온 Google에서 개발한 sentencepiece를 사용하고자 한다.

학습을 시킬 때 coverage는 기본값인 0.99995 보다 작은 최솟값인 0.98을 사용했다.
짧은 문장으로 구성되어 있고, 이번 프로젝트의 목적은 많이 나오는 채팅과 다르게 시청자에게 반응을 하기 위한
질문을 탐지해서 분류할 수 있는지에 대한 것이 목표이므로 '채팅스러운' 것을 찾으면 된다.
희귀하게 나오는 일반적인 단어들을 탐지하는 것이 아닌 일반적인 채팅을 탐지하고,
여집합인 소위 스트리밍을 시청하지 않는 일반인들이 보고 이해할 수 있는 문장을 분류하는 것이 좋다고 판단했기 때문이다.

스트리밍 채팅 속에서 '일반적인 질문'은 돌연변이 같은 존재라고 보고 접근을 하는 것이다.
'''

import sentencepiece as spm
from utils import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

DATA_FOLDER_NAME = 'preprocessed'
DATA_FOLDER = FILE_PATH(DATA_FOLDER_NAME)
MERGED_TXT_name = "merged_{streamer}.txt"
FORCE_REMAKE = True

def getStreamers():
    '''
    파일 경로가 다름으로 재정의 했음 지우지 말것
    '''
    for streamer in os.listdir(FILE_PATH(DATA_FOLDER_NAME)):
        if os.path.isdir(FILE_PATH(DATA_FOLDER_NAME, streamer)):
            yield streamer

def moveFileTo(this : os.PathLike, to : os.PathLike):
    if not os.path.exists(this):
        raise FileNotFoundError(this)
    os.rename(this, to)
    logging.info(f"Moved model and vocab file to {to}")

'''
input : 학습시킬 파일
model_prefix : 만들어질 모델 이름
vocab_size : 단어 집합의 크기
model_type : 사용할 모델 (unigram(default), bpe, char, word)
max_sentence_length: 문장의 최대 길이

pad_id, pad_piece: pad token id, 값
unk_id, unk_piece: unknown token id, 값
bos_id, bos_piece: begin of sentence token id, 값
eos_id, eos_piece: end of sequence token id, 값
user_defined_symbols: 사용자 정의 토큰
'''

def trainSentencePiece(streamer, model : str, vocabSize : int):
    COVERAGE = 0.98

    model_name = f"sp_{streamer}_{model}_{vocabSize}_{COVERAGE}"

    if already_exists(
        FILE_PATH(DATA_FOLDER_NAME, streamer, model,f"{model_name}.model"),
        FILE_PATH(DATA_FOLDER_NAME, streamer, model,f"{model_name}.vocab")
    ) and not FORCE_REMAKE:
        logging.info(f"already exists : {streamer}, {model}, {vocabSize}")
        return

    logging.info(f"start training : {streamer}, {model}, {vocabSize}")

    inputfile = FILE_PATH(DATA_FOLDER_NAME, streamer, MERGED_TXT_name.format(streamer=streamer))

    # --normalization_rule_name=identity
    # --num_threads=몇 개?
    formatInput = f'--input={inputfile} --model_prefix={model_name} --shuffle_input_sentence=True \
        --character_coverage={COVERAGE} --vocab_size={vocabSize} --model_type={model} --max_sentence_length=9999 --minloglevel=2'

    spm.SentencePieceTrainer.Train(formatInput)

    modelFile = f"{model_name}.model"
    vocabFile = f"{model_name}.vocab"

    if not os.path.isdir(FILE_PATH(DATA_FOLDER_NAME, streamer, model)):
        os.mkdir(FILE_PATH(DATA_FOLDER_NAME, streamer, model))

    if os.path.exists(FILE_PATH(DATA_FOLDER_NAME, streamer,model,modelFile)):
        os.remove(FILE_PATH(DATA_FOLDER_NAME, streamer,model,modelFile))
    if os.path.exists(FILE_PATH(DATA_FOLDER_NAME, streamer,model,vocabFile)):
        os.remove(FILE_PATH(DATA_FOLDER_NAME, streamer,model,vocabFile))

    moveFileTo(FILE_PATH(modelFile), FILE_PATH(DATA_FOLDER_NAME, streamer,model,modelFile))
    moveFileTo(FILE_PATH(vocabFile), FILE_PATH(DATA_FOLDER_NAME, streamer,model,vocabFile))

    return model_name

def encodeByWordpiece(streamer, modelName : str ,smp):

    modelName = modelName if modelName.endswith('.model') else f'{modelName}.model'
    # SentencePiece에서 경로가 str 파일이 아닐 경우 __str__ 하지 않고 에러 튕기는 문제 대문에 str()
    modelPath = str(FILE_PATH(DATA_FOLDER_NAME, streamer, modelName))

    sp = spm.SentencePieceProcessor()
    # logging.info(modelPath)
    sp.load(str(modelPath))

    t1 = sp.EncodeAsPieces(smp)
    t2 = sp.EncodeAsIds(smp)
    
if __name__ == '__main__':

    First = True
    Second = True

    if First:
        models = ['unigram','bpe','char','word']
        vocabSizes = [1000, 2000, 4000]
        sts = [x for x in getStreamers()]

        jobs_todo = len(models) * len(vocabSizes)

        from multiprocessing import Pool
        from itertools import product
        from os import cpu_count
        
        streamers_, models_, vocabSizes_ = zip(*list(product(sts, models, vocabSizes)))
        # logging.info(combinations)

        # 프로세스라서 CPU 개수(하이퍼스레딩 - 가상코어 포함을 의미)는 상관 없지만,  I/O가 많은 작업이고,
        # csv 파일이 클 경우에 시스템 전체에 블로킹을 일으킬 수 있으므로 전체 CPU 중 75%만 할당한다
        # 또한 sentencepeice는 모델을 변형하지 않고, 단순히 메모리에 적재 후 사용한다는 점.
        # 그리고 기존 파일을 읽기만 하고, 새로운 파일을 쓰는 것이기 때문에 상관 없다 (Lock 필요 X)

        # 나는 분명히 프로세서 16개 중 75%인 12개만 줬는데 한 순간에 28개의 프로세스가 돌아감
        # 뭐지?
        # pool = Pool(int(cpu_count() * 0.75))
        pool = Pool(4)
        pool.starmap(trainSentencePiece, product(sts, models, vocabSizes))

        # cnt = 0
        # for streamer in getStreamers():
        #     for model in models:
        #         logging.info(f'Working on {streamer.rjust(15)}, model : [{model}]')
        #         for vocabSize in vocabSizes:
        #             works = f'{str(cnt).rjust(2)}/{len(jobs_todo)}'.rjust(10)
        #             logging.info(f"streamer {streamer.rjust(15)} | {cnt}/{jobs_todo} ({works} %)")
        #             model_name = trainSentencePiece(streamer,model, vocabSize)
        #             cnt += 1
    
    if Second:
        pass
