# runpod-ai-lab

RunPod Ubuntu + NVIDIA CUDA에서 텐서 연산부터 LLM까지 직접 실험하는 Python/PyTorch 학습 저장소입니다. 초기 코드는 데이터 다운로드 없이 독립 실행됩니다. 설명은 한국어, 코드 식별자는 영어입니다.

## 구현 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 00 | 드라이버·CUDA·GPU 확인, 실제 행렬곱 | 구현 |
| 01 | 텐서 shape/stride/broadcasting, 행렬곱 성능 | 구현 |
| 02 | 2층 신경망, 수동 역전파·SGD, autograd 비교 | 구현 |
| 03 | PyTorch MLP 학습·검증·추론, 가중치 저장 | 구현 |
| 04 | MNIST / CNN | 계획 |
| 05 | Attention / Transformer | 계획 |
| 06 | Hugging Face 모델 | 계획 |
| 07 | LLM 추론 / 양자화 | 계획 |
| 08 | LoRA / QLoRA | 계획 |
| 09 | Stable Diffusion | 계획 |
| 10 | FastAPI 추론 서버 | 계획 |
| 11 | 동시 요청 / Batching / 성능 | 계획 |

각 단계 설명은 `labs/`에 있습니다. 04–11에는 실행 코드가 없으며 완료로 간주하지 않습니다.

CPU 실행·미분·학습 테스트를 통과했습니다. 정확한 환경과 GPU 미검증 범위는 [VALIDATION.md](VALIDATION.md)에 기록했습니다.

## RunPod에서 시작하기

1. RunPod 콘솔에서 NVIDIA GPU Pod를 생성합니다. Ubuntu + CUDA + PyTorch가 포함된 공식 PyTorch 템플릿을 선택하고 이미지 태그를 기록합니다. 초기 실습은 대형 GPU를 요구하지 않습니다. Python 3.10 이상, PyTorch 2.4 이상 3 미만을 사용합니다. 모델별 GPU 선택은 후속 실습 구현 시 별도로 검증합니다.
2. SSH 공개키를 RunPod 계정에 등록합니다. 키가 없다면 **로컬 컴퓨터**에서 `ssh-keygen -t ed25519`로 생성하고 `.pub` 내용만 등록합니다. 개인키는 업로드하지 않습니다.
3. Pod의 **Connect**에서 표시하는 SSH 명령을 로컬 터미널에 그대로 실행합니다. 호스트·포트·사용자는 Pod마다 다릅니다. 아래는 직접 TCP 연결 형식일 뿐 실제 값을 대입해야 합니다.

   ```bash
   ssh -i ~/.ssh/id_ed25519 root@<POD_PUBLIC_IP> -p <MAPPED_SSH_PORT>
   ```

   프록시 SSH를 쓰는 경우 Connect가 제공하는 별도 명령을 사용합니다. 파일 전송에는 직접 TCP SSH가 편리합니다. [RunPod SSH 안내](https://docs.runpod.io/pods/connect-to-a-pod)
4. 작업 경로를 영속 볼륨으로 설정합니다. 기본 마운트는 `/workspace`지만 콘솔에서 실제 경로를 확인하세요. 컨테이너 디스크, Pod 볼륨, 네트워크 볼륨의 보존 조건은 다릅니다. 네트워크 볼륨은 Pod와 독립적이며 Pod 볼륨은 Pod 삭제 시 보존되지 않습니다. [스토리지 안내](https://docs.runpod.io/pods/storage/types)
5. GitHub에 이 디렉터리를 게시했다면 아래처럼 설치합니다. `mix-hub`는 본인 계정으로 바꿉니다. 아직 게시하지 않았다면 이 디렉터리를 `/workspace/runpod-ai-lab`에 전송한 뒤 `cd`부터 실행합니다.

   ```bash
   cd /workspace
   git clone https://github.com/mix-hub/runpod-ai-lab.git
   cd runpod-ai-lab
   python3 --version
   nvidia-smi
   # 공식 템플릿의 CUDA PyTorch를 재사용합니다.
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   python -m pip install -e .
   python -m pip check
   source scripts/env.sh
   python -m runpod_ai_lab.environment --device cuda
   python -m runpod_ai_lab.tensors --device cuda
   python -m runpod_ai_lab.backprop --device cuda
   python -m runpod_ai_lab.train_mlp --device cuda
   ```

템플릿에 CUDA용 PyTorch가 없다면 독립 venv를 만들고 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 Linux / Pip / Python과 드라이버가 지원하는 CUDA 버전을 선택해 제공 명령을 먼저 실행하세요. CUDA wheel 인덱스를 프로젝트 의존성에 고정하지 않았습니다. `nvidia-smi`의 CUDA 표시는 드라이버 지원 상한이며 `torch.version.cuda`는 PyTorch 빌드 버전입니다. 시스템 CUDA toolkit 버전과 같을 필요는 없습니다.

`python3 -m venv`가 없다면 Ubuntu에서 `apt-get update` 후 `apt-get install -y python3-venv git`를 실행합니다. CPU 점검은 모든 명령의 `--device cuda`를 `--device cpu`로 바꾸면 됩니다. `auto`는 GPU가 없으면 CPU로 전환하므로 RunPod 검증에는 명시적 `cuda`를 권장합니다.

## 반복 실험

저장소 루트에서 실행합니다. 각 명령은 앞 실습의 산출물이 필요 없습니다.

```bash
python -m runpod_ai_lab.tensors --size 2048 --dtype float32 --repeats 50 --device cuda
python -m runpod_ai_lab.backprop --hidden 32 --lr 0.05 --steps 1000 --device cuda
python -m runpod_ai_lab.train_mlp --hidden 64 --depth 3 --batch-size 512 --epochs 50 --lr 0.003 --device cuda
# 다른 SSH 창에서 전체 GPU 상태 관찰
watch -n 1 nvidia-smi
# 동일 환경 복구를 돕기 위한 버전 기록 (runs는 Git 제외)
mkdir -p runs
python -m pip freeze > runs/environment.txt
python -m unittest discover -s tests -v
```

모든 실습은 `--seed`, `--device`, `--output`을 받습니다. `--help`로 옵션을 확인하세요. 한 번에 하나의 변수를 바꾸고 `runs/<UTC시각>_<실습>/metrics.json`을 비교하세요. 학습 실습은 같은 경로에 `model.pt`도 저장합니다. 같은 seed라도 GPU·PyTorch 버전·연산 구현에 따라 결과는 달라질 수 있습니다.

`allocated_mib`는 현재 텐서 메모리, `reserved_mib`는 PyTorch allocator 예약 메모리, `peak_allocated_mib`는 프로세스 실습 구간 최대 할당량입니다. 드라이버·CUDA context·다른 프로세스 메모리를 포함하는 `nvidia-smi`와 다릅니다. CPU 실행에서는 VRAM 필드가 null입니다. 타이머는 CUDA 동기화를 수행하며 텐서 실습에는 warmup이 있습니다. MLP 시간에는 첫 실행 초기화 영향이 포함될 수 있고, 데이터는 미리 GPU에 올리므로 데이터 로더 성능을 측정하지 않습니다. 작은 모델은 GPU가 CPU보다 느릴 수 있습니다.

## 구조와 확장

```text
labs/00_environment ... 11_performance/  단계별 설명과 개발 계획
src/runpod_ai_lab/                      실행 가능한 모듈과 공통 계측
scripts/env.sh                         캐시 경로 설정 (source로 실행)
tests/                                 수동 미분·학습·CLI 검증
runs/                                  자동 생성 결과·가중치 (Git 제외)
data/, .cache/                         후속 데이터셋·캐시 (Git 제외)
```

새 실습은 독립 모듈과 단계 README를 추가하고 데이터 출처, 실행 명령, 측정 지표, 최소 GPU 실측값, 검증 결과를 기록합니다. 후속 라이브러리는 해당 실습 구현 시 별도 선택 의존성으로 추가합니다. 현재 transformers, diffusers, FastAPI는 설치하지 않습니다.

모델 복원 예시 (`train_mlp` 결과 경로 사용):

```python
import torch
from runpod_ai_lab.train_mlp import build_model
checkpoint = torch.load("runs/<실험>/model.pt", map_location="cpu", weights_only=True)
model = build_model(checkpoint["hidden"], checkpoint["depth"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
with torch.inference_mode():
    print(model(torch.tensor([[0.5, -0.2]])).softmax(-1))
```

체크포인트는 추론용 가중치이며 optimizer 상태를 포함한 학습 재개는 아직 구현하지 않았습니다.

## 문제 해결과 저장소 관리

- CUDA unavailable: `nvidia-smi` → `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 순서로 확인합니다. CPU wheel 또는 GPU 미할당 여부를 확인하세요.
- OOM: `--size`, `--batch-size`, `--hidden`, `--depth`를 줄이고 다른 프로세스의 VRAM을 확인합니다. `empty_cache()`는 살아 있는 텐서를 해제하지 않습니다.
- 손실 발산: 학습률을 낮추고 seed를 고정합니다. 수동 역전파의 gradient 비교 실패는 그냥 무시하지 마세요.
- 긴 학습은 `tmux` 세션에서 실행하면 SSH 연결 종료에도 계속 실행할 수 있습니다.
- `.gitignore`는 데이터·캐시·가중치·결과·환경파일을 제외합니다. 사용자 지정 `--output` 경로는 직접 ignore에 추가하세요. `git add` 후 `git status`로 확인합니다.
- 사용 후 Pod를 중지하거나 종료하고 필요한 결과를 별도 보관하세요. GPU 중지 후에도 스토리지 비용이 발생할 수 있으므로 콘솔에서 확인합니다.

GitHub 저장소: https://github.com/mix-hub/runpod-ai-lab (비공개). clone/push에는 접근 가능한 GitHub 계정 인증이 필요합니다. 별도 계정에 복제하여 게시할 때는 원격 주소를 변경하세요. 새 로컬 저장소에서 게시하는 일반 절차:

```bash
git init -b main
git add .
git status
git commit -m "Add runnable GPU learning labs"
git remote add origin https://github.com/mix-hub/runpod-ai-lab.git
git push -u origin main
```
