import requests
from server.KEYS import CLIENT_ID, CLIENT_PW

HEADERS = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Naver-Client-Id': CLIENT_ID,
    'X-Naver-Client-Secret': CLIENT_PW,
}
str1="demo trasnslation of bbox1"

def translation_en2ko(sentence):
    data = f'source=en&target=ko&text={sentence}'.encode('utf-8')
    response = requests.post('https://openapi.naver.com/v1/papago/n2mt', headers=HEADERS, data=data)
    response_json=response.json()
    if response.status_code==200:
        return response.status_code, response_json['message']['result']['translatedText']
    else :
        return response.status_code, "papago..fail..."