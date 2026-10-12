# runpod-ai-lab

RunPod의 GPU, CUDA, PyTorch, Jupyter가 **이미 준비된 상태**에서 사용하는 AI 실습 저장소.

파일은 루트에만 둔다.

- `environment.py`: GPU와 PyTorch 환경을 한 번 확인하는 독립 스크립트
- `01_tensors.ipynb`: Tensor, GPU 행렬곱, CPU/GPU 시간과 VRAM
- `02_autograd.ipynb`: 직접 미분, autograd, 가중치 갱신
- `03_mlp_training.ipynb`: 신경망 학습과 추론
- `04_mnist_cnn.ipynb`: MNIST 55,000/5,000/10,000 분할, MLP·CNN 학습·비교·예측
- `05_image_generation.ipynb`: FLUX.1-dev 이미지 생성(4090 24GB, CPU offload), 프롬프트·시드·step 비교, latent·VRAM·시간 관찰
- `06_yolo_detection.ipynb`: YOLO11 객체 탐지, 사진·영상, 클래스·신뢰도·좌표, 출력 이미지 및 속도 비교

## 사용법

1. RunPod 웹터미널에서 저장소 디렉터리로 이동해 `python environment.py` 실행.
2. RunPod JupyterLab에서 `.ipynb` 파일을 열어 셀을 순서대로 실행.
3. 노트북은 각각 독립적으로 실행된다. Python 커널에 CUDA 지원 PyTorch가 있는지 확인.
4. 새로운 실습은 `07_주제.ipynb`, `08_주제.ipynb`처럼 번호만 이어서 추가.

프로젝트 등록, 별도 공통 모듈, CI는 사용하지 않는다. 01–03은 기존 PyTorch만 사용한다.
04번 실습은 PyTorch 외에 `torchvision`, `matplotlib`가 사전에 설치되어 있어야 하며, MNIST 최초 다운로드에는 인터넷 연결이 필요하다.
05는 Hugging Face에서 FLUX.1-dev 모델 접근 조건에 동의한 뒤, Diffusers 형식 가중치·설정 전체를 `/workspace/models/FLUX.1-dev/`에 다운로드한 상태를 전제로 한다. 노트북은 서버 로그인 없이 로컬 파일만 로딩한다. Diffusers 등의 라이브러리는 노트북 첫 부분의 설치 명령을 따른다. 4090 24GB는 sequential CPU offload를 기본값으로 사용하므로 CPU RAM과 실행 시간이 많이 필요할 수 있다.
05를 열고 위에서부터 실행하면 이미지가 셀 아래에 표시되고 `runs/05_image_generation/`에 저장된다.
06은 Jupyter에서 `%pip install ultralytics` 실행 후 셀을 순서대로 실행한다. 자체 예제 사진과 업로드한 사진·영상으로 물체를 탐지하며, 결과는 `runs/06_yolo_detection/`에 저장된다.
모델 캐시는 `.cache/`에 저장된다. 두 경로는 Git에서 제외되지만 노트북에 표시된 출력은 포함될 수 있으므로
커밋 전에 **Clear All Outputs** 후 저장한다. 05의 실제 RunPod GPU 생성은 아직 미검증이다.

기존 RunPod 서버에 이전 버전 저장소가 있다면 **저장소 디렉터리에서** `git pull`을 실행한다. `.venv`는 Git 관리 대상이 아니므로 유지된다.
