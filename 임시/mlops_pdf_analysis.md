# MLOps 아이디어 PDF 분석 및 프로젝트 적용 현황 리포트

## 📄 1. PDF 문서 핵심 요약
이 문서는 **"비구조화된 텍스트 데이터 환경에서 LLM 요약 모델이 겪는 데이터 드리프트를 어떻게 탐지하고, 환각(Hallucination) 없이 품질을 평가할 것인가"**에 대한 차세대 LLMOps 아키텍처 제안서입니다.

1. **무감독 데이터 드리프트 탐지**: 정답지가 없는 환경에서 입력 데이터의 분포 변화(공변량 시프트 등)를 찾아내기 위해 고차원 벡터 거리(MMD, 코사인 유사도)와 언어학적 지표(퍼플렉서티, 버스티니스), 그리고 통계적 검정(KS Test, PSI, 바세르슈타인 거리)을 사용합니다.
2. **사실성(Factual Consistency) 평가**: 단순 단어 매칭(ROUGE)이나 참조 무관 지표의 한계를 지적하며, **역추적 질의응답(QAFactEval)**과 **연속 가중 기댓값 기반의 G-Eval**, 그리고 사내 로컬망을 위한 **Prometheus** 모델의 도입을 제안합니다.
3. **LLMOps 자가 순환 아키텍처**: Evidently AI/NannyML로 드리프트를 감지하고, MLflow AI Gateway로 트래픽을 제어하며, 임계치(예: PSI > 0.25 또는 점수 < 72점)를 위반하면 ZenML을 통해 합성 데이터를 섞어 자동으로 재학습(Retraining)하고 Blue-Green으로 배포하는 완전 자동화 파이프라인을 구상했습니다.

---

## ✅ 2. 프로젝트에 충실히 접목된 요소 (Applied)

현재 `ML-HR` 프로젝트는 PDF의 핵심 사상(DataOps -> ModelOps -> Evaluation -> CT)을 코드 레벨에서 매우 충실하게 구현해 두었습니다.

* **Evidently AI 기반 드리프트 차단 (`collect.py`)**:
  * PDF에서 제시한 `PSI 스코어 0.25 초과 시 격리` 로직이 `detect_drift_for_topic` 함수와 조건문(`if drift_detected and psi_score > 0.25:`)으로 완벽히 구현되어 있습니다.
* **QAFactEval & G-Eval 연합 평가 (`advanced_eval.py`)**:
  * 단순 ROUGE 지표를 버리고, 프롬프트 기반의 `run_qafacteval`(사실성 팩트체크)과 `run_geval`(4대 루브릭 평가) 함수를 만들어 가중 평균(`get_comprehensive_score`)을 내는 첨단 방식을 적용했습니다.
  * G-Eval에서 PDF가 강조한 `Logprobs` 추출 기능(`logprobs=True, top_logprobs=2`)도 코드에 반영되어 있습니다.
* **데이터 레이크 및 AI Gateway 연동**:
  * `lakefs_snapshot` 테이블을 Supabase에 구성하여 원본 스냅샷을 보존하고 있습니다.
  * `LiteLLM`을 활용해 평가 트래픽을 통제하는 AI Gateway 역할을 수행 중입니다.
* **MLflow Tracing 도입 (`evaluate.py`, `advanced_eval.py`)**:
  * `@mlflow.trace` 데코레이터를 통해 PDF에서 언급한 'Nested Trace Trees(세부 실행 추적)' 레이턴시 로깅을 충실히 모사하고 있습니다.

---

## ❌ 3. 아직 적용되지 않은/축소된 요소 (Not Applied / Simulated)

현실적인 컴퓨팅 자원(GPU, 메모리)과 시간의 한계로 인해, PDF의 일부 거대 담론은 시뮬레이션으로 대체되거나 추후 과제로 남겨졌습니다.

* **초거대 로컬 LLM 및 Prometheus 미적용**:
  * PDF는 `Qwen3-30B`나 `GPT-OSS-120B` 같은 무거운 척추 모델과 오프라인 평가용 `Prometheus`를 명시했으나, 현재 프로젝트는 가벼운 로컬 모델(`KoBART`, `KoT5`, `RoBERTa`)과 클라우드 API(`GPT-4o-mini`)를 사용 중입니다. (Prometheus 연동은 `README.md`에 남은 과제로 기재됨)
* **심화 통계학적 검정 및 NannyML 부재**:
  * `Evidently AI`를 통한 PSI 검증은 구현되었으나, 텍스트 특징량(퍼플렉서티, 버스티니스) 계산이나 NannyML을 통한 `바세르슈타인 거리(Wasserstein Distance)`, `도메인 분류기(ROC AUC)` 등의 딥러닝 기반 무감독 탐지는 코드로 구현되지 않았습니다.
* **진짜 재학습(CT)의 생략 (시뮬레이션 처리)**:
  * 성능이 65점 미만으로 떨어졌을 때 `Optuna`와 `ZenML`을 통한 자동 파인튜닝 파이프라인이 가동되어야 하지만, 컴퓨터 과부하 방지를 위해 현재 `main.py`에서는 `# retrain_pipeline(topic=t)`처럼 **주석 처리(비활성화)**되어 경고 메시지만 출력하도록 축소 구현되었습니다.
  * 이에 따라 합성 데이터(Synthetic Data) 생성 및 Blue-Green 무중단 배포 로직도 현재는 빠져 있습니다.
