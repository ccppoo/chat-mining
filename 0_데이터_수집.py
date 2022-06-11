# -*- coding: utf-8 -*-
# 0_데이터_수집.py

'''
실시간 스트리밍 중 발생한 채팅 데이터를 수집하기에 수량을 충분히 확보하기 어렵고
제한적인 탓에 스트리밍이 끝난 회차의 전체 채팅 데이터를 "다시 보기" 영상을 통해서 다운로드한다.

"다시 보기"란 스트리밍을 다시 볼 수 있는 것을 의미하며, 스트리밍 당시 실시간으로 작성된 채팅 데이터 또한 불러 올 수 있기 때문에
실제 구하고자 하는 데이터와 똑같으므로 문제가 없다.

이 작업은 트위치 서버와 통신을 통해서 받는 작업이기 때문에 컴퓨팅 자원의 유무와 상관없이
동시에 많은 커넥션을 만들면 서버측으로부터 차단을 당할 수 있다.

그래서 많은 동시에 2개의 스트리밍 또는 하나씩 순차적으로 다운로드 받아야한다.
이 같은 점에 들어서 시간상 많이 걸리는 작업이다.
'''

from utils import *
import subprocess
import logging
from secret import CLIENT_ID, CLIENT_SECRET,streamers

logger = logging.getLogger()

output_folder = partial(pathlib.Path, os.getcwd())

TIMEZONE = "Asia/Seoul"
FORMAT = "csv"
NUM= 20

cmd = "python -m tcd \
    --client-id {CLIENT_ID} --client-secret {CLIENT_SECRET} \
    --channel {streamer} --format {format} \
    --first={num} --output {ppath} \
    --timezone {timezone}"

cmd = "python -m tcd \
    --client-id {} --client-secret {} \
    --channel {} --format {} \
    --first={} --output {} \
    --timezone {}"

def run_subprocess(streamer):
    if not os.path.exists(output_folder('data')):
        os.makedirs(output_folder('data'))
    if not os.path.exists(output_folder('data', streamer)):
        os.makedirs(output_folder('data', streamer))
        
    cmdReady = cmd.format(CLIENT_ID, CLIENT_SECRET, streamer,FORMAT, NUM, output_folder('data', streamer), TIMEZONE)

    subprocess.run(cmdReady, shell=True, check=True)

if __name__ == '__main__':
    '''
    병렬로 작업할 경우라도 트위치 서버에서 접속량 많으면 차단해버림
    다른 IP를 가진 PC를 이용해서 병렬화하던가 프록시를 사용하던가 해야한다.
    '''
    for streamer in streamers:
        run_subprocess(streamer)