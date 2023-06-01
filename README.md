# NSFW Detector 

유해사진 검출 프로젝트 입니다.
 Dockerfile로 생성된 Docker 이미지는 AWS ECR을 통해 AWS Lambda에 사용 가능하도록 제작되었습니다. Docker로 바로 실행 가능합니다.


### model description
> 모델 사용은 nsfw_model.py를 참고하세요.

- 실사 검출용 original model
- 애니메이션 Fine-tuning한 model

### model link
- open_nsfw_model [original link](https://drive.google.com/file/d/1PXtPzLN3EVFHZTmMHfy0KeaNjWK7ZP2r/view?usp=share_link)
- Fine-tuning extra animation image based on open_nsfw [fine-tuning model line](https://drive.google.com/file/d/1YXSt7OxUqJG9uBmhLrM2eO3WZk32Ubuy/view?usp=share_link)