import datetime
from dateutil.tz import gettz
from slack_sdk import WebClient
from utils.utils import *

class SlackAPI:
    """
    슬랙 API 핸들러
    """
    def __init__(self, token):
        # 슬랙 클라이언트 인스턴스 생성
        self.client = WebClient(token)
        
    def get_channel_id(self, channel_name):
        """
        슬랙 채널ID 조회
        """
        # conversations_list() 메서드 호출
        result = self.client.conversations_list()
        # 채널 정보 딕셔너리 리스트
        channels = result.data['channels']
        # 채널 명이 'test'인 채널 딕셔너리 쿼리
        channel = list(filter(lambda c: c["name"] == channel_name, channels))[0]
        # 채널ID 파싱
        channel_id = channel["id"]
        return channel_id

    def get_message_ts(self, channel_id, query):
        """
        슬랙 채널 내 메세지 조회
        """
        # conversations_history() 메서드 호출
        result = self.client.conversations_history(channel=channel_id)
        # 채널 내 메세지 정보 딕셔너리 리스트
        messages = result.data['messages']
        # 채널 내 메세지가 query와 일치하는 메세지 딕셔너리 쿼리
        message = list(filter(lambda m: m["text"]==query, messages))[0]
        # 해당 메세지ts 파싱
        message_ts = message["ts"]
        return message_ts

    def post_thread_message(self, channel_id, message_ts, text):
        """
        슬랙 채널 내 메세지의 Thread에 댓글 달기
        """
        # chat_postMessage() 메서드 호출
        result = self.client.chat_postMessage(
            channel=channel_id,
            text = "학습 중!",
            thread_ts = message_ts,
            blocks=[
                {
                    'type': 'section',
                    'text': {
                        'type': "mrkdwn",
                        'text': text
                 }
                }
            ]
        )
        return result

def get_init_message(device, batch_size, val_ratio, num_workers, model, optimizer, loss_fn, num_epoch):
    message = f"""
    `Model 학습 시작`
    • 시각: {datetime.datetime.now(gettz('Asia/Seoul')).strftime("%Y/%m/%d, %H:%M:%S")}
    • device: {device}

    `Data Settings`
    • BATCH_SIZE = {batch_size}
    • Train : Validation = {(1-val_ratio)*100}% : {val_ratio*100}%
    • NUM_WORKERS = {num_workers} # dataloader

    `Model Settings`
    • Total params: {(count_parameters(model, True) + count_parameters(model, False)):,}
    • Trainable params: {(count_parameters(model, True)):,}
    • Non-trainable params: {(count_parameters(model, False)):,}
        
    `Optimizer Settings`
    • Optimizer
    {optimizer.__repr__()}

    `Training Settings`
    • NUM_EPOCH = {num_epoch}
    • LOSS = {loss_fn.__repr__()}
    """
    return message

def get_epoch_message(phase, epoch, epoch_loss, epoch_mask_f1, epoch_gender_f1, epoch_age_f1, epoch_final_f1, epoch_final_precision, epoch_final_recall, epoch_final_accuracy):
    message = f"""
    • phase: `{phase}`
    • 현재 epoch: {epoch}
    • 평균 Loss: {epoch_loss:.5f}
    • 평균 F1-score(mask)   : {epoch_mask_f1:.5f}
    • 평균 F1-score(gender)   : {epoch_gender_f1:.5f}
    • 평균 F1-score(age)   : {epoch_age_f1:.5f}
    • 평균 F1-score(total)   : {epoch_final_f1:.5f}
    • 평균 Precision(total)   : {epoch_final_precision:.5f}
    • 평균 Recall(total)   : {epoch_final_recall:.5f}
    • 평균 Accuracy(total)   : {epoch_final_accuracy:.5f}
    """
    return message

def get_final_message():
    message = f'''
    `모델 학습 종료!` 
    • 종료 시각: {datetime.datetime.now(gettz('Asia/Seoul')).strftime("%Y/%m/%d, %H:%M:%S")}
    '''
    return message