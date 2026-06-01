## 개발 환경 및 라이브러리 버전 (Prerequisites)

- OS : Windows 11 (64-bit)
- Python : 3.13.x
- CUDA : 12.x 기반 (PyTorch 연동)

## 주요 라이브러리 버전:

- torch : 2.x.x+cu12x (GPU 연산용)
- transformers : 4.38.2 (Hugging Face 모델 허브)
- supabase : 2.x.x (데이터 적재용)
- mlflow : 2.x.x (실험 추적 및 지표 기록)
- openai : 1.x.x (GPT-4o-mini 평가용)

## 프로젝트 아키텍쳐

- Pipeline Flow:
  - Naver News API -> collect.py(수집) -> Supabase DB -> auto_pipeline.py(3-Model 요약) -> evaluate.py(하이브리드 채점) -> DB 업데이트

## 핵심 기능

- Multi-Model Benchmarking : KoBART, KoT5, KoBART-v3 총 3개 모델의 요약 성능 비교
- Hybrid AI Evaluation : GPT-4o-mini, HyperCLOVA X, 코사인 유사도를 결합한 하이브리드 채점 시스템
- Automated Scheduling : 8시간 주기로 뉴스를 자동 수집하고 요약하여 데이터 드리프트에 선제적 대응
- Champion Model Selection : 분야별 최근 20건 평균 점수로 최적 모델을 자동 선정

## 설치 및 실행 방법 (로컬 환경)

1. **환경 변수 파일 세팅 (`.env`)**
   - 프로젝트 최상단에 `.env` 파일을 생성하고 다음 키들을 입력합니다.
     ```env
     SUPABASE_URL=당신의_URL
     SUPABASE_KEY=당신의_KEY
     NAVER_CLIENT_ID=네이버_아이디
     NAVER_CLIENT_SECRET=네이버_시크릿
     HF_TOKEN=허깅페이스_토큰
     OPENAI_API_KEY=GPT_평가용_키
     CLOVA_API_KEY=클로바_키(선택사항)
     ```

2. **필수 라이브러리 설치**
   ```powershell
   pip install -r requirements.txt
   ```

3. **서비스 실행** (터미널을 각각 열어 실행)
   - **MLflow 서버 (실험 관리)**:
     ```powershell
     mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
     ```
   - **FastAPI 서버 (모델 서빙)**:
     ```powershell
     uvicorn src.server:app --host 0.0.0.0 --port 8000
     ```
   - **Streamlit 대시보드 (모니터링)**:
     ```powershell
     streamlit run src/app.py --server.port=8501
     ```
   - **자동화 스케줄러 (데이터 수집 및 파이프라인)**:
     ```powershell
     python src/main.py
     ```

4. **서비스 접속 주소**
   - 📊 **Streamlit 대시보드 (모니터링)**: `http://localhost:8501`
   - ⚡ **FastAPI 서버 (API 문서)**: `http://localhost:8000/docs`
   - 📈 **MLflow UI (실험 추적 및 지표)**: `http://localhost:5000`

# 프로젝트 진행 상황

## 완료된 내용

- **[DataOps] 실시간 뉴스 수집** : 네이버 뉴스 API를 통해 8개 분야(IT, 경제, 사회, 정치, 연예, 스포츠, 생활/문화, 세계)의 최신 뉴스를 자동으로 가져오는 파이프라인 완성 (collect.py)
- **[ModelOps] 멀티 모델 요약** : KoBART, KoT5, KoBART-v3 총 3개 모델을 기준으로 요약 결과 비교 구조 구현 (auto_pipeline.py)
- **[Evaluation] 하이브리드 채점 시스템** : ROUGE 방식을 폐기하고, GPT-4o-mini + HyperCLOVA X + 수학적 코사인 유사도(ko-sroberta)를 결합한 가중 앙상블 채점으로 전면 교체 (evaluate.py)
- **[Database] 데이터베이스 통합** : Supabase를 연동하여 원문 데이터, 모델별 요약본, 채점 결과를 실시간으로 적재 및 관리 (news_data, model_stats 테이블)
- **[CI/CD] 자동화 파이프라인** : GitHub Actions를 통한 8시간 주기 자동 파이프라인 구축 및 로컬 schedule 라이브러리 연동 (main.py, mlops_cron.yml)
- **[MLOps] 실험 관리** : MLflow Tracking Server를 연동하여 모델별 하이브리드 점수를 분야×모델 단위로 연속 기록
- **[DevOps] API 서빙** : FastAPI를 이용한 Champion 모델 조회, 파이프라인 수동 트리거, CT(Continuous Training) 시뮬레이션 엔드포인트 구축 (server.py)
- **[Monitoring] 대시보드** : Streamlit 기반 분야별 랭킹, 드리프트 상태 모니터링, 모델별 요약 비교 대시보드 구현 (app.py)
- **[자동화] 모델 선발 규칙** : 분야별 최근 20건 평균 점수 기반 Champion 모델 자동 선정 및 성능 하락(65점 미만) 감지 로직 구현

## 아직 시작하지 않은 내용

- Evidently AI를 활용한 통계 기반 데이터 드리프트 분석
- Docker화 및 클라우드(Render/AWS) 배포
- 실제 모델 재학습(Fine-tuning) 파이프라인 연동

## 한계점 및 향후 과제

- **QAFactEval / G-Eval 미구현** : 논문 기반의 학술 평가 지표(QAFactEval, G-Eval) 도입을 목표했으나, 한국어 환경에서의 전용 QA 모델 부재 및 logprobs 가중합 구현의 복잡도로 인해 현재 LLM-as-Judge 방식으로 대체
- **CT(Continuous Training) 시뮬레이션** : 재학습 엔드포인트는 구현되어 있으나 실제 모델 파인튜닝은 수행하지 않으며, MLflow 메트릭 로깅 시뮬레이션으로 동작
- **KoT5 안정성** : pko-t5-base 모델의 토크나이저 호환성 이슈로 간헐적 에러 발생 가능
