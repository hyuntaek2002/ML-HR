#  ML-HR 프로젝트 프레젠테이션 기획안 (표준 목차 적용 & 코드 매핑)

보여주신 이전 팀의 프레젠테이션 목차(큰 틀)를 유지하면서, 현재 진행하신 **뉴스 요약 LLMOps 프로젝트**의 핵심 내용(Data Drift, Hallucination 평가, MLflow Tracing 등)을 완벽하게 매핑하고, **발표 시 참고할 수 있도록 실제 구현된 코드 파일과 주요 함수를 주석으로 추가**했습니다.

---

## 1. 서론 및 개요 (Introduction & Overview)
*이전 팀의 '전체적인 아키텍처, 일정, 문제 해결 방식' 파트*

* **프로젝트 소개**
  * 뉴스 요약 모델 벤치마킹 및 실시간 운영 파이프라인(LLMOps) 구축
* **핵심 문제 및 이슈**
  * **이슈 1 (Data Drift):** 뉴스는 신조어와 트렌드가 매우 빠르게 변해, 학습된 모델의 성능이 금방 저하됨.
  * **이슈 2 (Hallucination):** 요약 데이터는 명확한 '정답지(Reference)'가 없기 때문에 모델이 헛소리를 하는지 평가하기 어려움.
* **문제해결방식**
  * 다양한 관점의 3개 모델(KoBART, KoT5, RoBERTa) 벤치마킹
  * 무참조(Reference-free) 검증 및 데이터 드리프트 실시간 감지(Evidently AI) 파이프라인 구축
* **프로젝트 전체 일정 및 협업 규칙 설정**
  * DataOps 도구(Git, DVC, Supabase)를 활용한 데이터 및 코드 버전 관리 협업 방식

---

## 2. 데이터 및 모델 설명 (DataOps & ModelOps)
*이전 팀의 데이터 수집, 전처리, 사용 모델, 하이퍼파라미터 파트*

* **데이터 수집**
  * 실시간 크롤링: 네이버 뉴스 API
  * 학습 및 기준(Baseline) 데이터: AI-Hub 문서요약 텍스트
  > 💻 **코드 참고:** `src/collect.py` (네이버 뉴스 API 크롤러 및 Supabase 데이터 적재 로직)
* **전처리 및 변수 구성**
  * 텍스트 정제(특수문자, 불용어 처리), 모델별 전용 토크나이저 적용, 입력 길이(Token Limit) 제한
  > 💻 **코드 참고:** `src/preprocess.py` (문장 정제 및 전처리 함수)
* **어떤 모델을 사용했는가? (3 Multi-Model)**
  * **KoBART:** 경량성과 연산 효율성 중심
  * **KoT5:** 자연스러운 문장 생성 능력 중심
  * **KLUE-RoBERTa:** 핵심 문장 추출 성능 비교용 (추출형 요약)
  > 💻 **코드 참고:** `src/evaluate.py` (3개 모델의 추론 및 요약 생성 로직)
* **[LLMOps 특화] 데이터 드리프트 탐지 및 통제**
  * Evidently AI와 PSI 스코어를 활용한 가독성 분석.
  * 기준치(PSI > 0.25) 초과 시 이상 데이터를 원천 격리하는 로직 구현.
  > 💻 **코드 참고:** `src/data_drift.py` (Evidently AI 및 PSI 계산 로직), `src/collect.py` (수집 중 드리프트 감지 시 차단하는 `detect_drift_for_topic` 적용 부분)

---

## 3. 실험 및 평가 결과 (Evaluation & MLflow)
*이전 팀의 MLflow 성능 지표 실험, 결과 및 모델 선정 파트*

* **새로운 성능 평가 지표 (정답 없는 평가의 혁신)**
  * 기존 단어 매칭(ROUGE) 방식의 한계를 넘어선 정밀 채점 도입
  * **QAFactEval:** 기계 독해 기반 사실성(Fact) 검증
  * **G-Eval:** LLM-as-a-judge(로그 확률 Logprobs 기반) 4대 루브릭 평가
  > 💻 **코드 참고:** `src/advanced_eval.py` (`run_qafacteval`, `run_geval`, `get_comprehensive_score` 함수)
* **MLflow 기반 실험 및 지표 관리**
  * **MLflow Tracing:** 모든 파이프라인 모듈에 추적 데코레이터를 붙여 레이턴시 및 프롬프트 모니터링 연동 (Nested Trace Trees)
  > 💻 **코드 참고:** `src/evaluate.py` 및 `src/advanced_eval.py` 내의 `@mlflow.trace` 데코레이터 적용 부분, `src/sync_to_mlflow.py`
* **AI Gateway 구축 (비용 통제)**
  * LiteLLM을 활용하여 평가(Evaluator) API 트래픽을 중앙 집중화하고 비용을 통제
  > 💻 **코드 참고:** `src/advanced_eval.py` (LiteLLM 라우팅 및 LLM API 호출 부분)
* **결과 및 모델 최종 선정**
  * 3개 모델의 레이턴시, 비용, QAFactEval/G-Eval 점수를 종합하여 최적의 서빙 모델을 선정하는 기준 제시

---

## 4. 배포 및 운영전략 (Deployment & Observability)
*이전 팀의 CI/CD, Docker, 파이프라인 흐름도 파트*

* **전체 파이프라인 흐름도**
  * News API ➡️ Crawler ➡️ Supabase DB ➡️ 3-Model Inference & Evaluator ➡️ Update
  > 💻 **코드 참고:** `src/auto_pipeline.py` (전체 흐름을 오케스트레이션 하는 로직)
* **모니터링 및 서빙 인프라**
  * **FastAPI:** 모델 서빙을 위한 REST API
  * **Streamlit:** 수집, 평가, 드리프트 상태를 실시간 확인하는 모니터링 대시보드
  > 💻 **코드 참고:** `src/server.py` (FastAPI 서버), `src/app.py` (Streamlit 대시보드 화면 구성)
* **자동화 및 CT (Continuous Training) 파이프라인**
  * 6시간 주기 뉴스 자동 수집 및 요약 스케줄러
  * **CT 시뮬레이션:** 드리프트 감지 시 격리 데이터+합성 데이터를 활용해 파인튜닝을 트리거하고 Blue-Green 배포 플래그를 세우는 ZenML 기반 로직 (현재는 자원 한계로 알림 시뮬레이션 적용)
  > 💻 **코드 참고:** `src/main.py` (크론 스케줄링 및 CT 트리거 `retrain_pipeline()` 주석 처리된 시뮬레이션 부분)
* **Docker 컨테이너화 및 배포**
  * Docker를 통한 실행 환경 표준화 및 Render(또는 로컬) 기반 배포 환경 구성

---

## 5. 결론 및 향후 개선 (Conclusion & Future Work)
*이전 팀의 한계 및 느낀점 파트*

* **성과 요약**
  * 단순한 요약 모델 개발을 넘어, 성능 저하(Drift)를 모니터링하고 모델을 지속적으로 평가하는 '차세대 LLMOps 아키텍처' 완성
* **한계 및 느낀점 (시사점)**
  * PDF 논문의 방대한 아이디어를 실제 코드로 구현하면서 겪은 컴퓨팅 자원(GPU, 메모리)의 한계와, 이를 시뮬레이션으로 유연하게 대처한 경험
* **향후 개선 과제**
  * 로컬 GPU(Prometheus 모델) 기반 오프라인 평가 완전 자동화 연동
  * AWS EC2 등 클라우드 인스턴스로 파이프라인 마이그레이션 및 서비스화
