# 스마트 제조 RxM — LLM 멀티에이전트 조율 수준 비교 실험

C-MAPSS(터보팬) 데이터로 회귀(RUL)·분류(고장임박)·이상탐지를 수행하는
하이브리드 MAS. 커맨더(Claude Opus 4.8)의 조율 수준 L1~L4를 바꿔가며
**안정성·비용**을 비교한다. LangGraph 기반.

---

## 1. 사전 준비 (서버)

- **Python 3.10+**
- **Ollama** 설치 후 실행 (SLM 구동) — https://ollama.com
  - GPU 권장: `qwen3:8b`는 약 6.6GB VRAM 필요 (부족하면 CPU로 느려짐)
- **Anthropic API 키** (커맨더용, 인터넷 연결 필요)

## 2. 설치

### 2-1. 파이썬 패키지
```bash
pip install -r requirements.txt
```

### 2-2. Ollama 설치 + 서버 띄우기 (SLM 구동)
```bash
# 설치 — systemd 서비스로 자동 등록·실행됨 (리눅스 서버)
curl -fsSL https://ollama.com/install.sh | sh

# 서버 떠 있는지 확인
systemctl status ollama          # active (running) 이면 OK
#   서비스가 안 뜨거나 컨테이너 환경이면 수동 실행(백그라운드):
#   nohup ollama serve > ollama.log 2>&1 &

# SLM 모델 받기
ollama pull qwen3:8b

# 확인
ollama list                      # qwen3:8b 보이면 OK
ollama ps                        # 떠 있는 모델 + GPU/CPU 비율 (GPU여야 빠름)
nvidia-smi                       # GPU 인식·메모리 확인
```
※ Ollama 서버가 떠 있어야 `llm.py`의 SLM 호출이 동작한다 (기본 포트 11434).

## 3. API 키 설정 (.env)

프로젝트 폴더에 `.env` 파일 만들고:
```
ANTHROPIC_API_KEY=sk-ant-여기에_실제_키
```
(`.env`는 `.gitignore`에 있어 깃에 안 올라감)

## 4. 데이터 확인

`data/` 폴더에 C-MAPSS FD001 파일 3개가 있어야 함:
```
data/train_FD001.txt
data/test_FD001.txt
data/RUL_FD001.txt
```

## 5. 연결 확인

```bash
python verify_api.py          # 커맨더(Claude) 연결 확인
python llm.py                 # 커맨더 + SLM 둘 다 확인
```
둘 다 응답 나오면 준비 완료.

## 6. 한 번 실행 (단일 런 테스트)

```bash
python graph.py <레벨> <태스크> --run

# 예시
python graph.py L1 regression --run
python graph.py L4 anomaly --run
```
- 레벨: `L1` `L2` `L3` `L4`
- 태스크: `regression` `classification` `anomaly`

→ 완주여부·점수·모델·처방분포·호출수가 출력됨.

## 7. 배치 실험 (전체 런)  ※ runner.py 작성 후

```bash
python runner.py              # 방식 A: 4레벨×3태스크×N회
```
→ 결과는 `logs/runs.jsonl`에 한 줄씩 누적.

---

## 폴더 구조

```
rxm_experiment/
├── config.py          # 설정·요청문장·임계값·정책
├── prompts.py         # 시스템 프롬프트 8개 (전 레벨 고정)
├── state.py           # 공유 상태(State) 정의
├── llm.py             # Claude / Ollama 호출
├── graph.py           # LangGraph 조립 (L1~L4)
├── logger.py          # 런 결과 → JSONL
├── runner.py          # 배치 실행 (작성 예정)
├── analyze.py         # 안정성·비용 집계 (작성 예정)
├── data/
│   ├── data_prep.py   # 데이터 준비 (RUL 라벨·3태스크)
│   └── *_FD001.txt    # C-MAPSS 원본
└── nodes/
    ├── commander.py   # 커맨더 4행동 (route/guide/review/control_ml)
    ├── perception.py / preprocessing.py / analysis.py / prescription.py
    ├── _parse.py      # SLM JSON 파싱
    └── _ctx.py        # 커맨더 지침 주입 헬퍼
```

## 모델 구성
- 커맨더(LLM): `claude-opus-4-8` (외부 API) — `config.COMMANDER_MODEL`
- 에이전트(SLM): `qwen3:8b` (로컬 Ollama) — `config.SLM_MODEL`
