# runpod-ai-lab

RunPod의 GPU, CUDA, PyTorch, Jupyter가 **이미 준비된 상태**에서 사용하는 AI 실습 저장소.

파일은 루트에만 둔다.

- `environment.py`: GPU와 PyTorch 환경을 한 번 확인하는 독립 스크립트
- `01_tensors.ipynb`: Tensor, GPU 행렬곱, CPU/GPU 시간과 VRAM
- `02_autograd.ipynb`: 직접 미분, autograd, 가중치 갱신
- `03_mlp_training.ipynb`: 신경망 학습과 추론
- `04_mnist_cnn.ipynb`: MNIST 55,000/5,000/10,000 분할, MLP·CNN 학습·비교·예측

## 사용법

1. RunPod 웹터미널에서 저장소 디렉터리로 이동해 `python environment.py` 실행.
2. RunPod JupyterLab에서 `.ipynb` 파일을 열어 셀을 순서대로 실행.
3. 노트북은 각각 독립적으로 실행된다. Python 커널에 CUDA 지원 PyTorch가 있는지 확인.
4. 새로운 실습은 `05_주제.ipynb`, `06_주제.ipynb`처럼 번호만 이어서 추가.

패키지 설치, 프로젝트 등록, 별도 공통 모듈, CI는 사용하지 않는다. 04번 실습은 PyTorch 외에 `torchvision`, `matplotlib`가 사전에 설치되어 있어야 하며, MNIST 최초 다운로드에는 인터넷 연결이 필요하다.

기존 RunPod 서버에 이전 버전 저장소가 있다면 **저장소 디렉터리에서** `git pull`을 실행한다. `.venv`는 Git 관리 대상이 아니므로 유지된다.
