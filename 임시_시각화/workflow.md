# ML-HR MLOps 파이프라인 구동 흐름도

이 문서는 ML-HR 프로젝트의 전체적인 시스템 아키텍처와 데이터 흐름(DataOps -> ModelOps -> Evaluation -> CT/DevOps)을 시각화한 다이어그램입니다.

```mermaid
graph TD
    %% 스타일 정의
    classDef dataOps fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef modelOps fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef evalOps fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef devOps fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef database fill:#eceff1,stroke:#455a64,stroke-width:2px,stroke-dasharray: 5 5;
    
    %% 외부 소스
    NaverAPI(("Naver News API"))

    %% DataOps 영역
    subgraph DataOps ["1. 데이터 수집 및 전처리 (DataOps)"]
        A1["collect.py<br/>8개 분야 텍스트 수집"]
        A2["데이터 정제 & HTML 세탁"]
        A3{"Evidently AI<br/>실시간 가독성/Drift 검사"}
        A4["이상 데이터 격리"]
    end

    %% 데이터베이스
    DB[("Supabase DB<br/>news_data 테이블")]:::database
    Lake[("lakeFS<br/>스냅샷 보존")]:::database

    %% ModelOps 영역
    subgraph ModelOps ["2. 다중 모델 요약 (ModelOps)"]
        B1["auto_pipeline.py<br/>미처리 뉴스 조회"]
        B2["KoBART 추론"]
        B3["KoT5 추론"]
        B4["KLUE-RoBERTa 추론"]
    end

    %% Evaluation 영역
    subgraph Evaluation ["3. AI 심사위원 평가 (Evaluation)"]
        C1["evaluate.py<br/>MLflow Tracing"]
        C2["LiteLLM AI Gateway<br/>트래픽/비용 통제"]
        C3["GPT-4o-mini & HyperCLOVA X<br/>QAFactEval / G-Eval 검증"]
        C4["MLflow Server<br/>메트릭 로깅"]
    end

    %% DevOps / CT 영역
    subgraph DevOps ["4. 스케줄링 및 재학습 (CT & DevOps)"]
        D1["main.py<br/>6시간 주기 크론잡"]
        D2{"분야별 1위 모델<br/>평균 점수 < 65점?"}
        D3["ZenML / Optuna<br/>자동 파인튜닝 트리거"]
        D4["Streamlit Dashboard<br/>app.py"]
    end

    %% 흐름 연결
    NaverAPI --> A1
    A1 --> A2
    A2 --> A3
    A3 -- "정상 (PSI < 0.25)" --> DB
    A3 -- "정상" --> Lake
    A3 -- "비정상 (Drift)" --> A4

    DB --> B1
    B1 --> B2 & B3 & B4
    B2 & B3 & B4 -- "요약문 DB 업데이트" --> DB

    DB --> C1
    C1 --> C2
    C2 --> C3
    C3 -- "평가 점수 DB 업데이트" --> DB
    C3 -- "지표 기록" --> C4

    D1 -->|트리거| A1
    D1 -->|트리거| B1
    D1 -->|트리거| C1
    DB -->|랭킹 집계| D2
    D2 -- "Yes (성능 열화)" --> D3
    D2 -- "No (정상)" --> D4
    DB -->|실시간 시각화| D4
    C4 -.-> D4

    %% 서브그래프 스타일 적용
    class A1,A2,A3,A4 dataOps;
    class B1,B2,B3,B4 modelOps;
    class C1,C2,C3,C4 evalOps;
    class D1,D2,D3,D4 devOps;
```

## 주요 파이프라인 단계 설명

1. **DataOps (수집 및 차단)**
   네이버 뉴스 API에서 데이터를 가져온 뒤, `Evidently AI`를 통해 데이터 분포나 가독성에 이상이 없는지(Drift) 실시간으로 검사합니다. 정상 데이터는 Supabase와 데이터 레이크(lakeFS)에 저장됩니다.

2. **ModelOps (추론 경쟁)**
   저장된 뉴스를 바탕으로 3개의 경량 로컬 모델(`KoBART`, `KoT5`, `RoBERTa`)이 각각 요약문을 생성하고 결과를 다시 데이터베이스에 적재합니다.

3. **Evaluation (자동 평가 및 검증)**
   생성된 요약문들을 상용 대형 LLM(GPT-4o-mini 등)이 평가합니다. 이때 `LiteLLM`으로 API 트래픽을 통제하며, `QAFactEval`(무참조 사실성 검증) 기법을 활용해 환각을 잡아내고 `MLflow`에 추적 로그를 남깁니다.

4. **CT & DevOps (지속적 학습 및 모니터링)**
   스케줄러(`main.py`)가 전체 과정을 주기적으로 통제하며, 특정 분야의 1위 모델 점수가 65점 밑으로 떨어지면 성능 열화로 판단하여 `ZenML`을 통한 자동 재학습(파인튜닝)을 트리거합니다. 사용자는 이 모든 과정을 `Streamlit` 대시보드에서 실시간으로 관제합니다.
