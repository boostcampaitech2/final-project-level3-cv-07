# :movie_camera: 오이(OCR_In_Editor)

### CV-07 컴퓨터구조 :computer: 
|[김성민](https://github.com/ksm0517)|[박지민](https://github.com/ddeokbboki-good)|[박진형](https://github.com/ppjh8263)|[심세령](https://github.com/seryoungshim17)|[윤하정](https://github.com/YHaJung)|
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://ifh.cc/g/puHQTP.jpg" width="120" height="120"> | <img src="https://user-images.githubusercontent.com/82632580/147220995-f83623c7-da26-404f-ad07-d3da19928e65.jpg" width="120" height="120">| <img src="https://user-images.githubusercontent.com/82632580/147216442-3d820ddd-7a04-4c1c-b717-0bab4b3aed0b.jpg" width="120" height="120"> |<img src="https://user-images.githubusercontent.com/82632580/147216194-d7100c74-e273-465a-815c-85f8032f4be0.png" width="120" height="120">| <img src="https://user-images.githubusercontent.com/82632580/147216286-e1e30025-9dac-4fa8-b931-cc663a2d0ca1.jpg" width="120" height="120">| 

## 💡Project 소개
- 편집 툴에 영상을 넣고 **"Send Current Frame"** 버튼을 눌러주면 선택한 프레임 속 **텍스트를 자동번역** 해주는 프로그램
- 모델이 구한 글자와 배경색을 같이 계산하여 최대한 영상에 자연스럽게 적용될 수 있도록 결과를 출력
- 번역된 자막을 **원하는 내용만 선택 적용**할 수 있어 영상 편집 시 유용하게 사용할 수 있음
- 효율적인 배포를 위해 **Github Action**과 **Docker**를 사용하여 **CI/CD**를 진행
### 🔎예시
![예시](https://user-images.githubusercontent.com/82632580/147242498-9c8da7ea-a159-491c-ac53-009616c47246.png)

## ⚙️Pipeline
![화면 캡처 2021-12-23 210528](https://user-images.githubusercontent.com/82632580/147240587-577c9408-8add-4cd5-b465-6dc6d665669e.png)
### Model 설정  
- 영상 편집이라는 상황에 맞게 inference가 빠르고 가벼운 모델 선정    
  
![화면 캡처 2021-12-23 212230](https://user-images.githubusercontent.com/82632580/147240612-5ec7ecf5-eee4-4f0c-a487-fd9f31f07abc.png) 
|종류|모델|
|:---:|:---:|
|Detector|**EAST**|
|Recognizer|**CRNN or RNet**|
### Openshot(영상편집툴) 
![open](https://user-images.githubusercontent.com/82632580/147244480-0eb298f9-64e7-4e7a-bf07-2d57e85002ae.png)  

### Server via FastAPI

![ser](https://user-images.githubusercontent.com/82632580/147244791-c9994d60-fa4a-4b01-875c-921605bac396.png)
1. 기능
    - 이미지 POST
    - OCR Model Inference
    - Get background & word color
    - Text Translate with Papago
2. CI & CD
    - Github Action과 Docker사용
    - CI & CD 결과를 즉각적으로 알 수 있게 Slack으로 결과 전송
    - base Docker Image 생성하여 재배포 시간 30초 이내 시행
