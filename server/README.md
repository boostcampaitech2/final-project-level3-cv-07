# OCR In Editor's Server via FastAPI
해당 프로젝트는 [부스트캠프 최종프로젝트](https://github.com/boostcampaitech2/final-project-level3-cv-07)를 위한 BackEnd 개발에 관한 내용입니다.

## Server 기능
1. 이미지 POST 
    - FastAPI를 사용해서 API 생성
    - 이미지를 받고 번역된 문장, Bounding Box, Word&Backgroud 색상을 Responese
    - [Swagger](http://www.realbro.site/docs)를 통해 API문서 관리
2. OCR Model Inference
    - [모델 팀](https://github.com/boostcampaitech2/final-project-level3-cv-07/tree/main/models)에서 연구 & 개발한 모델을 사용
    - API를 통해 받은 이미지를 Inference하여 Image내의 Text와 Bounding Box정보를 얻음
3. Get background & word color
    - Bounding Box내의 Image Histogram 정보를 사용, Background와 Word의 색상을 추출
    - API를 통해 받은 이미지를 Inference하여 Image내의 Text와 Bounding Box정보를 얻음
    - 기존 K-means를 사용하여 이미지의 색상을 추출하는 방법을 생각 했으나 \
    Response Time이 10~20초이상 걸리는 문제점이 생겨 Histgram 방식으로 변경, 5초 이내의 Response 가능
4. Text Translate with Papago
    - 최종 번역된 결과를 얻기 위해 모델에서 Inference한 Text를 Papago API를 통해 번역 향후 이 또한 번역 모델로 사용 가능할것으로 예상
    
    
## CI & CD
- 효율적인 CI & CD를 위해 Github Action과 Docker사용
- 이를 위해 개발 서버와 운용 서버의 분리 후 운용 서버 [도메인](http://realbro.site/docs)연결
- Main Brunch server folder 기준 Update시 ssh사용 서버 접속, pull한 후 Dockerize하여 배포
- 처음 Docker Image Build시에 필요한 모든 것들을 새로 설치하여 Image를 재생성하는 과정 때문에 10분 넘게 걸리는 문제점 존재 \
이후 Base Docker Image 생성하여 재배포 시간 30초 이내 시행
- Github Action을 통한 CI & CD의 결과를 즉각적으로 알 수 있도록 Slack으로 결과 전송

## Folder Structure

```
[server]
├── modules                     # From /Models/modules 
├── saved                       # Model Weight Folder
├── scripts                     # Shell Scripts Folder
├── server                      # Server's Source code
│   ├── api                     # Router Folder
│   ├── modules                 # Detail Function Folder
│   │   ├── inference.py        # Model Inference
│   │   ├── papgo.py            # Papago API
│   │   ├── color_finder.py     # Get Word&Backgrund Colors
│   │   └── ...
│   └── ...
├── api.py                      # Server Main Source code
├── Dockerfile                  # Docker File using at CI&CD Dockerize
├── pyproject.toml              # Poetry Config
├── poetry.lock                 # Poetry File from pyproject.toml
└── ...
```

## Prerequisites


```
cd server

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# Activate poetry to use
source ~/.poetry/env
poetry install
poetry run poe force-cuda11
poetry shell
```

## Running server

```
python api.py
```
