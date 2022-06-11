# -*- coding: utf-8 -*-
# 0_전처리.py

'''
스트리밍 중 발생한 데이터는 python 모듈인 tcd(twitch chat downloader)을 통해서 받았다.
최초로 받을 때 적용한 CSV 포멧은 아래와 같다.

파일 이름 : [stream id].csv
    * real time : 채팅이 발생한 절대적인 시각
    * uptime    : 방송 시작하고 경과한 시간을 기준으로 채팅이 발생한 시각
    * nickname  : 채팅을 친 유저의 닉네임(웹/앱 인터페이스 상으로 보이는 이름)
    * chat      : 채팅 원본

(1) stringify_comma

    최초의 데이터는 CSV 포멧으로 되어 있지만, 채팅 내부에 쉼표(',')가 있어 pandas dataframe을 사용하지 못하는 상태이므로
    아하 작성한 "stringify_comma" 함수를 이용해 원본 채팅은 그대로 살리기 위해 큰 따옴표로 둘러싸고
    원본 데이터가 있는 폴더 'data'가 아닌 'preprocessed/csv' 폴더에 csv 파일을 저장했다.

(2) is_system_chat

    그리고 실제 시청자가 아닌 시스템에서 보낸 안내성/이벤트성 메세지를 제거하기 위해서
    이하 작성한 "is_system_chat" 함수를 이용해 시스템 메세지일 경우 길이가 0 인 문자열을 반환해
    pandas dataframe에서 제거할 수 있도록 했다.

    시스템 메세지는 트위치에서 공식으로 제공하는 챗봇(NightBot) 외에 스트리머가 자유롭게 사용할 수 있는
    플러그인 형태의 챗봇이 존재하기 때문에 스트리머 개개인 별로 시스템 메세지를 필터링하기 위한 추후 작업이 필요하다.
    모든 스트리머에서 공통적으로 나타나는 시스템 챗봇(Nightbot)은 제거한 상태다.

(3) remove_space_and_special

    스트리밍 중 발생하는 채팅은 대부분 한글으로 이루어져있으며, 소량 영어로 된 채팅이 존재한다.
    영어로된 채팅의 경우 트위치 채팅 시스템 내 제공되는 이모티콘이 이모티콘 이름으로 변환되어 나타나는데 (예: funzinAng1)
    채팅이라고 간주하기에는 부적절하나, 'ㅋㅋ'와 같이 자음으로 자신의 감성을 표출하는 시청자와 외관상 다르지만
    같은 의미로 사용된다는 판단하에 제거하지 않기로 했다.

    필터링을 하기위해서 정규식을 사용했고 숫자/영어/한글/물음표 외에 특수문자, 한자, 등 언어는 제거했고
    이로 인해서 발생되는 공백은 한 칸의 공백으로(whitespace)으로 대체되었다.
    사용된 정규식 패턴은 다음과 같다 : '[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣?]'

(4) compress_repetitive_char, compress_repetitive_word

    스트리밍의 채팅에서는 채팅의 길이만큼 자신의 감정이 격하다는 것을 보여준다.
    이와 같은 특성 때문에 하이라이트 구간에서 'ㅋ'를 8~20 회 반복해서 작성하는 경우를 많이 볼 수 있다.
    이와 같이 반복되는 자소의 경우 임베딩과 가중치에 노이즈를 발생시킬것이 우려되어
    2회 이상 연속으로 반복되는 공백을 포함한 문자는 2회로 축소시켰다.

    연속된 하나의 문자('W')가 아닌 단어의 경우('WORLD') 단순히 반복되는 자소보다는
    의미가 더 있을거라고 간주하여 n회 이상 반복되었을 경우 n회(짝수일 경우 n, 홀수의 경우 n+1)로 축소시켰다.

(5) time_str_to_sec

    추후 벡터화 된 채팅을 클러스터로 구성했을 때 시간대에 따라 등장하는 채팅의 특성을 시각화 하기 위하여
    uptime을 초(second)으로 변환한 것이다.

(6) preprocess

    위에 서술한 1~5번 과정을 하나로 종합한 함수다.

(7) to_txt, txt_all_in_one

    전처리된 문장("preprocessed" column)을 토크나이저에 학습시키기 위한 파일을 미리 만들기 위해서
    한번의 스트림을 text 파일로 변환하고, 스트리밍 회차와 상관없이 보편적으로 등장하는 어휘와 토큰을 찾기 위해서
    수집된 자료들에 한해서 하나의 텍스트 파일로 모으는 txt_all_in_one 함수를 작성했다.

    이렇게 만들어진 텍스트 파일은 google에서 개발된 sentencepiece의 모델을 학습시키기 위해서 사용된다.

※   1회의 스트리밍에 평균적으로 2~3만 개의 채팅 텍스트, 많게는 5만 적게는 1~2만 개로 짧지만 많은 텍스트를 다루다보니
    시간 효율성을 위해서 Multiprocess module을 사용하고 있다.
    이 코드를 실행하는 환경에 따라 Multiprocess module 사용 여부를 바꾸면 된다.

* -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * --
| 원본 데이터                                         : ./data/<streamer id>/<stream id>.csv
| 전처리, 토큰화한 데이터                              : ./preprocessed/<streamer id>/csv/<stream id>.csv
| 텍스트 파일로 변환한 데이터                          : ./preprocessed/<streamer id>/txt/<stream id>.txt
| 모든 스트림을 하나의 텍스트 파일로 취합한 데이터      : ./preprocessed/<streamer id>/merged_<streamer id>.txt
* -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * -- * --
'''

from utils import *
from koreanCHARs import *
import re
import pandas as pd
from hanspell import spell_checker
from multiprocessing import Pool

HEADERS = ['real time', 'uptime', 'nickname', 'chat']

NUM_ENG_KOR_Q = '[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣?]'
KOR = '[^ㄱ-ㅎㅏ-ㅣ가-힣]'
MERGED = "MERGED_{}.txt"
FORCE_REMAKE = True

def stringify_comma(filePath : os.PathLike, saveAt : os.PathLike, as_name : str = None):
    '''
    데이터 원본에는 ['real time', 'uptime', 'nickname', 'chat'] 총 4개의 column이 있다.
    마지막 'chat' column의 경우 채팅 문자열 데이터 자체에 콤마(',')가 있어 csv 파일을 읽는데 문제가 있으므로
    채팅 문자열에 해당하는 부분은 큰 따옴표(" ... ")를 이용해서 감싼다.

    이렇게 처리한 CSV 파일은 폴더 preprocessed/<streamer id>/csv 에 똑같은 구조로 만든다.
    '''

    if not os.path.isdir(saveAt):
        os.makedirs(saveAt)

    if not os.path.isfile(filePath) or not os.path.exists(filePath):
        raise FileNotFoundError(filePath)

    if already_exists(pathlib.Path(saveAt, f'{as_name}.csv')):
        logging.info(f"passing file since already exists : {f'{as_name}.csv'}")
        return

    saves = []

    # logging.info(f"{filePath=}")
    with open(filePath, mode='r', encoding=UTF_8) as fpp:
        for line in fpp:
            target = line.strip().split(',', maxsplit=3)
            target[3] = ''.join(target[3:])
            target[3] = f'"{target[3]}"'
            saves.append(target)

    name = as_name if as_name else os.path.basename(filePath).split('.')[0]

    save_file_name = pathlib.Path(saveAt, f'{name}.csv')
    
    import csv
    
    with open(save_file_name, mode='w', newline='', encoding=UTF_8) as fp:
        csvwriter= csv.writer(fp)
        csvwriter.writerow(HEADERS)
        csvwriter.writerows(saves)
    return

def is_system_chat(sender : str, chat : str) -> bool :
    '''
    시스템 메세지는 스트리머가 메세지로 노출을 할 것인지 설정할 수 있지만,
    스트리머마다 노출을 할지 말지에 대한 여부를 선택하므로 모두 적용되지 않는다.

    챗봇, 알림 메세지, 등 다양한 봇 또한 존재하기 때문에 스트리머 각각 적용해야하는 점도 있으므로
    차후 프로젝트를 심화할 때 이를 보완한다. 

    아래 종류 이름 옆 (O/X)는 정규식을 통해서 필터링을 하는지에 대한 여부를 뜻한다.

    1) 구독메세지 (O)
        트위치에는 스트리머에게 정기적으로 후원하는 '구독(subscription)' 메세지가 있는데
        이는 지동으로 발생하는 시스템 메세지이고 의미가 없으므로 제거한다.

    2) 트윕, 후원 메세지 (X)
    3) 챗봇 메세지, 공지/알림 메세지 (X)
    '''

    sender = sender or ''
    chat = chat or ''

    BOTS = [
        'Nightbot',
    ],

    SPAMMER = []

    def isBOT(sender):
        if sender in BOTS:
            return True
        return False

    def isSPAMMER(sender):
        if sender in SPAMMER:
            return True
        return False

    def subscription(sender, chat):
        PATTERN = ".* subscribed (at|with) (Tier|Prime).*"
        # '피클호로록 gifted a Tier 1 sub to 침척맨'
        PATTERN2 = ".* gifted a Tier [0-9]+ sub to .*"
        # 피클호로록 is gifting 5 Tier 1 Subs to 김도 s community They ve gifted a total of 18 in the channel55
        PATTERN3 = ".* is gifting [0-9]+ Tier [0-9]+.*"
        PATTERN4 =".*여러분은 트위치 최고의 대가족을 찾아라 이벤트에 참여 중입니.*"
        if re.match(PATTERN, chat) or re.match(PATTERN2, chat) or re.match(PATTERN3, chat) or re.match(PATTERN4, chat):
            return True
        return False
    
    def twip(sender, chat):
        return False

    def chat_bot(sender, chat):
        return False
    
    return any([
        subscription(None, chat), 
        twip(None, chat),
        chat_bot(None, chat), 
        isBOT(sender), 
        isSPAMMER(sender)
    ])

def remove_space_and_special(string : str) -> str:
    '''
    물음표, 숫자, 영어, 한글 제외한 다른 특수 문자 제거를 한다.
    다른 특수문자를 제거한 뒤 공백 하나를 기준으로 다시 조합해 반환한다.

    예) AA??!_B@@B_C喝 --> AA?? B B C
    '''
    result = string.strip()
    result = ' '.join(re.sub(NUM_ENG_KOR_Q, ' ', result).split())
    return result

def compress_repetitive_char(string : str, n : int = 2) -> str:
    '''
    공백을 포함한 모든 문자를 대상으로 연속된 하나의 문자(char)가 n회 이상 반복되었을 경우
    n 회로 줄여서 반환한다.

    예) AAAA BBB_CC CC --> AA BB_CC CC
    '''
    r = ""
    length = len(string)
    
    if length < n:
        return string

    cnt = 1
    i = 1

    while i < length:
        if string[i] == string[i - 1]: 
            cnt += 1
        else:
            if cnt > 1:
                r += string[i - 1]*n
            else:
                r += string[i - 1]
            cnt = 1
        i += 1

    if cnt > 1:
        r += string[i - 1]*n
    else:
        r += string[i - 1]

    return r

def compress_repetitive_word(string : str, n : int = 2) -> str:
    '''
    문자열 내 반복이 되는 같은 단어가 있으면 줄임
    n번 이상 반복될 경우 n회로 축소

    예) haa haa haa --> haa haa
    '''
    answer = len(string)
    compressedss = []

    # 1개 단위(step)부터 압축 단위를 늘려가며 확인
    for step in range(1, len(string) // 2 + 1):
        compressed = ""
        prev = string[0:step]  # 앞에서부터 step만큼의 문자열 추출
        count = 1
    
        # 단위(step) 크기만큼 증가시키며 이전 문자열과 비교
        for j in range(step, len(string), step):
        
            # 이전 상태와 동일하다면 압축 횟수(count) 증가
            if prev == string[j:j + step]:
                count += 1
            
            # 다른 문자열이 나왔다면
            else:
                # compressed += str(count) + prev if count >= 2 else prev
                if count >= 2:
                    if len(prev) >= 2:
                        compressed += prev
                    elif len(prev) == 1:
                        compressed += prev * 2
                else: 
                    compressed += prev
                prev = string[j:j + step]  # 다시 초기화
                count = 1
            
        # 남아 있는 문자열에 대해서 처리
        # compressed += str(count) + prev if count >= 2 else prev
        if count >= 2:
            if len(prev) >= 2:
                compressed += prev
            elif len(prev) == 1:
                compressed += prev * 2
        else: 
            compressed += prev
        # 만들어지는 문자열이 가장 짧은 것이 정답
        compressedss.append(compressed)

    line = sorted({x : len(x) for x in compressedss}.items(), key=lambda x : x[1])

    if line:
        line = line[0] 
        if line[1] < answer:
            # logging.info(f"{line[0]} vs {s}")
            return line[0]
    return string

def time_str_to_sec(string_time):
    hour, minute, seconds = list(map(lambda x : int(x), string_time.split(':')))
    # logging.info(hour, minute, seconds)
    seconds = hour *60 *60 + minute*60 + seconds
    return seconds

def preprocess(streamer : str, pool : Pool):
    '''
    stringify_comma
    is_system_chat
    remove_space_and_special
    compress_repetitive_char
    compress_repetitive_word

    함수를 이용해서 csv 파일에 저장할 수 있도록 처리함
    '''
    
    CSVFILEs = [x for x in getFilesFrom('data', streamer, extension='csv')]
    SAVE_FOLDER = "preprocessed"

    def workTodo(i, path_, csvname):
        logging.info(f'[{streamer}] Working on {i}/{len(CSVFILEs)} ({i/len(CSVFILEs)*100:.2f}%) : file name : {csvname}')

        # logging.info(f'stringify_comma ... ')
        f = FILE_PATH(SAVE_FOLDER, streamer,'csv')

        if already_exists(FILE_PATH(PREPROCESSED, streamer,'csv', csvname)) and not FORCE_REMAKE:
            logging.info(f"passing csv file that already exists :: {csvname}")
            return

        stringify_comma(ORIGIN_FILE_PATH(streamer, csvname), f)

        df = None
        ff = FILE_PATH(SAVE_FOLDER, streamer,'csv', csvname)

        with open(ff, mode='r', encoding=UTF_8) as fp:
            df = pd.read_csv(fp)

        def apply_df(row, repeatlimit : int = 2) -> str:
            if is_system_chat(row['nickname'], row['chat']) == True:
                return ''
            string = remove_space_and_special(row['chat'])
            string = compress_repetitive_char(string, repeatlimit)
            string = compress_repetitive_word(string, repeatlimit)
            return f'"{string}"'

        def apply_hanspell(row, ) -> str:
            '''
            반복되는 문자나 단어는 문법과 상관없이 줄여야하는 것이므로 
            원본인 'chat'이 아닌 중복을 처리한 PREPROCESSED 열을 참조한다
            '''
            if not any(re.sub(KOR, ' ', row[PREPROCESSED]).replace(' ', '')):
                return row['chat']
            try:
                a = spell_checker.check(row[PREPROCESSED]).checked
            except:
                print(f'시발  : {row["chat"]}')
            else:
                return a

        logging.info(f'adding upsecond column for time series')
        df['upsecond'] = df.apply(lambda row : time_str_to_sec(row['uptime']), axis=1)

        logging.info(f'Compressing...')
        df[PREPROCESSED] = df.apply(lambda row : apply_df(row), axis=1)

        # logging.info(f'Applying Hanspell...')
        # df[HANSPELL] = df.apply(lambda row : apply_hanspell(row), axis=1)

        logging.info(f'Drop empty string ... ')
        df = df.drop(df[df.preprocessed == ''].index)

        logging.info(f'Saving at {ff}')
        # df = df[['real time','uptime', 'upsecond','nickname', 'chat', PREPROCESSED, HANSPELL]]
        df = df[['real time','uptime', 'upsecond','nickname', 'chat', PREPROCESSED]]
        with open(ff, mode='w', newline='', encoding='utf-8') as fp:
            df.to_csv(fp, index=False)

        # if not os.path.exists(SAVE_AT(streamer, ORIGIN)) or not os.path.isdir(SAVE_AT(streamer, ORIGIN)):
        #     os.makedirs(SAVE_AT(streamer, ORIGIN))

    # for i, (path_, csvname) in enumerate(getFilesFrom('data', streamer, extension='csv'), start=1):
    #     workTodo(i, path_, csvname)

    a = [(i, path_, csvname) for i, (path_, csvname) in enumerate(getFilesFrom('data', streamer, extension='csv'), start=1)]

    pool.starmap(workTodo, a)

def to_txt(streamer):
    # txt 파일만 지원하는 다른 것들을 위해서 txt 파일로 만들기
    if not os.path.isdir(FILE_PATH(PREPROCESSED, streamer, 'txt')):
        os.mkdir(FILE_PATH(PREPROCESSED, streamer, 'txt'))

    for dir_path, csvfile in getFilesFrom(PREPROCESSED, streamer, 'csv', extension='csv'):
        
        f_path = pathlib.Path(dir_path, csvfile)
        filename = csvfile.split('.')[0]

        if already_exists(FILE_PATH(PREPROCESSED, streamer,'txt', f'{filename}.txt')) and not FORCE_REMAKE:
            logging.info(f"passing csv file that already exists :: {filename}.txt")
            continue

        df = pd.read_csv(f_path)
        chats = [chat.strip('""')+'\n' for chat in df[PREPROCESSED].to_list()]

        with open(FILE_PATH(PREPROCESSED, streamer, 'txt', f'{filename}.txt'), mode='w', encoding='utf-8') as fp:
            fp.writelines(chats)
    return

def txt_all_in_one(streamer):
    if not os.path.isdir(FILE_PATH(PREPROCESSED, streamer)):
        os.mkdir(FILE_PATH(PREPROCESSED, streamer))

    # merging all text files to single txt file
    merged_txt_path = FILE_PATH(PREPROCESSED, streamer, f'merged_{streamer}.txt')
    logging.info(f'Merging text files in one text file ... to {merged_txt_path}')

    chats = []
    # logging.info(list(getFilesFrom(PREPROCESSED, streamer, 'txt', extension='txt')))

    for dir_path, txtfile in getFilesFrom(PREPROCESSED, streamer, 'txt', extension='txt'):
        with open(FILE_PATH(dir_path , txtfile), mode='r', encoding='utf-8') as fp:
            while(line := fp.readline()):
                chats.append(line)

    with open(merged_txt_path, mode='w', encoding='utf-8') as fp:
        fp.writelines(chats)

if __name__ == '__main__':
    from multiprocessing import Pool

    sts = [x for x in getStreamers()]

    pool = Pool(4)
    # pool.map(preprocess, sts)

    for streamer in sts:
        logging.info(f'starting streamer data : {streamer}')
        preprocess(streamer)
        to_txt(streamer)
        txt_all_in_one(streamer)