## 개발 환경 및 라이브러리 버전 (Prerequisites)

- OS : Windows 11 (64-bit)
- Python : 3.13.x
- CUDA : 12.x 기반 (PyTorch 연동)

## 주요 라이브러리 버전:

- torch : 2.x.x+cu12x (GPU 연산용)
- transformers : 4.x.x (Hugging Face 모델 허브)
- supabase : 2.x.x (데이터 적재용)
- mlflow : 2.x.x (실험 추적 및 지표 기록)
- rouge-score : 0.1.2 (성능 평가 지표)

## 프로젝트 아키텍쳐

- Pipeline Flow:
- Naver News API -> Python Crawler(collect.py) -> Supabase DB -> 3-Model Inference(evaluate.py) -> Summary Update


## 핵심 기능

- Multi-Model Benchmarking : 서로 다른 아키텍처를 가진 3개 모델의 요약 성능 비교
- Automated Scheduling : 6시간 주기로 뉴스를 자동 수집하고 요약하여 데이터 드리프트에 선제적 대응
- Baseline Comparison : 생성형 요약 모델과 추출형 기준 모델을 함께 비교

## 설치 및 실행 방법 (로컬 환경)

백지 상태의 새로운 노트북(PC) 환경에서도 다음 단계만 거치면 전체 파이프라인이 즉시 가동됩니다.

1. **패키지 매니저 및 가상환경 세팅**
   - `uv` 등 패키지 매니저를 사용하여 가상환경을 생성하고 의존성을 설치합니다.
   ```powershell
   uv venv --python 3.12 .venv
   .\.venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. **환경 변수 파일 세팅 (`.env`)**
   - 프로젝트 최상단에 `.env` 파일을 생성하고 다음 키들을 입력합니다.
     ```env
     SUPABASE_URL=당신의_URL
     SUPABASE_KEY=당신의_KEY
     NAVER_CLIENT_ID=네이버_아이디
     NAVER_CLIENT_SECRET=네이버_시크릿
     HF_TOKEN=허깅페이스_토큰
     OPENAI_API_KEY=GPT_평가용_키
     # CLOVA_API_KEY=선택사항
     ```

3. **서비스 개별 실행**
   - 터미널 창을 4개 열고, 각 창마다 가상환경을 활성화(`.\.venv\Scripts\activate`)한 뒤 다음 명령어들을 각각 실행합니다:

   - **MLflow 서버 (실험 관리)**:
     ```powershell
     mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
     ```
     mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow_personal.db --default-artifact-root ./mlartifacts_personal

   - **FastAPI 서버 (모델 서빙)**:
     ```powershell
     uvicorn src.server:app --host 0.0.0.0 --port 8000
     ```
   - **Streamlit 대시보드 (모니터링)**:
     ```powershell
     streamlit run src/app.py --server.port=8501
     ```
   - **스케줄러 (데이터 수집 및 크론잡)**:
     ```powershell
     python src/main.py
     ```

4. **서비스 접속 주소**
   - 📊 **Streamlit 대시보드 (모니터링)**: `http://localhost:8501`
   - ⚡ **FastAPI 서버 (재학습 트리거/문서)**: `http://localhost:8000/docs`
   - 📈 **MLflow UI (실행 추적 및 지표)**: `http://localhost:5000`

# LLMOps 고도화 진행 상황 (V2)

## 최근 적용 완료된 내용 (차세대 파이프라인)

- **[DataOps] 통계 기반 실시간 차단**: Evidently AI와 PSI 기반 가독성 분석을 통해 이상 데이터 원천 격리 (data_drift.py)
- **[ModelOps] AI Gateway 구축**: LiteLLM을 활용하여 평가(Evaluator) API 트래픽 중앙 집중화 및 비용 통제
- **[Evaluation] 환각 방지 무참조 검증**: QAFactEval(기계 독해 기반) 및 G-Eval(로그 확률 기반)을 활용한 정밀한 AI 채점 도입
- **[Observability] MLflow Tracing**: 모든 파이프라인 모듈에 추적 데코레이터를 붙여 레이턴시 및 프롬프트 모니터링 연동
- **[CT] ZenML 재학습 자동화 트리거**: 이상 감지 시 격리 데이터+합성 데이터를 활용한 파인튜닝 트리거 및 Blue-Green 배포 플래그 로직 구성

## 앞으로 해볼 수 있는 남은 과제
- 로컬 GPU(Prometheus 모델) 오프라인 평가 완전 자동화 연동
- AWS EC2 등 클라우드 인스턴스에 파이프라인 배포 및 서비스화
