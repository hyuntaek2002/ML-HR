## 개발 환경 및 라이브러리 버전 (Prerequisites)

- OS : Windows 11 (64-bit)
- Python : 3.13.x
- CUDA : 12.x 기반 (PyTorch 연동)

## 주요 라이브러리 버전:

- torch : 2.x.x+cu12x (GPU 연산용)
- transformers : 4.x.x (Hugging Face 모델 허브)
- bitsandbytes : 0.4x.x (Mistral-7B 4-bit 양자화용)
- supabase : 2.x.x (데이터 적재용)
- mlflow : 2.x.x (실험 추적 및 지표 기록)
- rouge-score : 0.1.2 (성능 평가 지표)

## 프로젝트 아키텍쳐

- Pipeline Flow:
- Naver News API -> Python Crawler(collect.py) -> Supabase DB -> 4-Model Inference(evaluate.py) -> Summary Update

## 핵심 기능

- Multi-Model Benchmarking : 서로 다른 아키텍처를 가진 4개 모델의 요약 성능 실시간 비교
- Automated Scheduling : 6시간 주기로 뉴스를 자동 수집하고 요약하여 데이터 드리프트에 선제적 대응
- Low-Resource Optimization : 4-bit 양자화를 통해 소비자용 GPU에서 7B급 LLM 구동

## 설치 및 실행 방법

- `.env` 파일에 네이버 API와 Supabase 키 설정
- 필수 라이브러리 설치: `pip install -r requirements.txt`
- 공식 실행 진입점: `python src/main.py`
- 배치 실행: `.\run_all.bat`

## 현재 기준 파이프라인

- 공식 파이프라인: `collect.py -> evaluate.py`
- `collect.py`: 네이버 뉴스 수집 및 Supabase 저장
- `evaluate.py`: 4개 모델 요약 생성 및 모델별 컬럼 저장
- `main.py`: 공식 실행 엔트리포인트
- `auto_pipeline.py`: 단일 KoBART 데모 스크립트로, 메인 파이프라인에는 포함하지 않음

## `news_data` 기준 스키마

- `id`: 뉴스 레코드 기본 키
- `title`: 기사 제목
- `description`: 기사 본문 또는 대체 본문
- `originallink`: 원문 링크
- `topic`: 뉴스 카테고리
- `summary_kobart`: KoBART 요약 결과
- `summary_kot5`: KoT5 요약 결과
- `summary_roberta`: KLUE-RoBERTa 추출 결과
- `summary_mistral`: Mistral-7B 요약 결과

현재 기준에서는 모델별 결과 저장을 표준으로 사용합니다.
기존 `summary` 컬럼은 `auto_pipeline.py` 전용 레거시 컬럼이며, 메인 파이프라인에서는 사용하지 않습니다.

# 현재 프로젝트 진행 상황 점검

## 완료된 내용

- 실시간 DataOps 구축 : 네이버 뉴스 API를 통해 8개 분야의 최신 뉴스를 자동으로 가져오는 파이프라인 완성 (collect.py)
- 멀티 모델 서빙 환경 : KoBART, KoT5, KLUE-RoBERTa, Mistral-7B 총 4개 모델을 로컬 GPU 환경에서 4-bit 양자화 등을 활용해 구동 성공
- 데이터베이스 통합 : Supabase를 연동하여 원문 데이터와 모델별 요약본을 실시간으로 적재 및 관리 (news_data 테이블)
- 자동화 파이프라인 : 수집과 요약을 한 번에 실행하는 배치 스크립트(run_all.bat) 및 6시간 주기 자동화(Windows 스케줄러 설정) 체계 마련
- 실험 관리 기초 : MLflow를 연동하여 모델별 성능(ROUGE, Latency)을 기록할 준비 완료

## 진행 중 / 해결해야 할 내용

- KoT5 에러 해결 : 현재 DB에 Error로 찍히는 T5 모델의 환경 설정(Tokenizer/Library) 최적화
- 평가 로직 구체화 : ROUGE 점수가 0점이 나오지 않도록 정답지를 제목이 아닌 핵심 문장으로 변경하거나 의미 유사도 도입
- 인사고과 대시보드 : DB에 쌓인 모델별 결과물을 한눈에 비교할 수 있는 시각화 레이아웃 구상

## 아직 시작하지 않은 내용

- DevOps (배치 -> API) : FastAPI를 이용해 가장 성적이 좋은 모델을 외부에서 호출할 수 있게 만드는 작업
- Monitoring : Evidently AI를 활용한 데이터 드리프트 분석
- Deployment : Docker 라이징 및 클라우드(Render/AWS) 배포
- GPT-3.5 구현
