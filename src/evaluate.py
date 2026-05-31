import os
import re
import json
import requests
import time
import torch # 🚨 수학적 유사도 연산을 위한 파이토치
from transformers import AutoTokenizer, AutoModel # 🚨 임베딩 추출을 위한 트랜스포머
from dotenv import load_dotenv
from supabase import create_client
import mlflow  # 🚨 MLflow 라이브러리 연동
import litellm # 🚨 AI Gateway 역할을 수행할 LiteLLM 연동
import openai

# 환경 설정 및 초기화
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from advanced_eval import get_comprehensive_score

# 수학적 유사도 모델 및 토크나이저 전역 캐싱 (매번 로드하여 OOM 나는 현상 방지)
_math_tokenizer = None
_math_model = None

def get_gpt_score(summary_text, original_text):
    """지표 1: GPT-4o-mini 심사위원을 통한 요약 품질 채점"""
    prompt = f"원문: {original_text}\n요약: {summary_text}\n품질을 0~100점 사이 JSON으로 평가해: {{\"score\": 0}}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            timeout=15
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0))
    except Exception as e:
        print(f"(GPT 에러: {e}) ", end="")
        return None

def get_clova_score(summary_text, original_text):
    """지표 2: Clova HyperCLOVA X 심사위원을 통한 요약 품질 채점"""
    url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    headers = {
        "Authorization": f"Bearer {os.getenv('CLOVA_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": f"뉴스 원문과 요약문을 비교하여 요약 품질을 0~100 사이의 숫자로만 평가해줘.\n\n원문: {original_text}\n요약: {summary_text}"}],
        "topP": 0.8, "topK": 0, "maxTokens": 256, "temperature": 0.1, "stopBefore": [], "repeatPenalty": 1.1
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['result']['message']['content']
            score_match = re.search(r'\d+(\.\d+)?', content)
            if score_match:
                return float(score_match.group())
        return None
    except Exception as e:
        print(f"(Clova 에러: {e}) ", end="")
        return None

def get_math_similarity_score(summary_text, original_text):
    """지표 4: 100% 수학적 선형대수학 기반 고차원 문장 임베딩 코사인 유사도 연산 (BERTScore 스타일)"""
    global _math_tokenizer, _math_model
    try:
        if _math_tokenizer is None or _math_model is None:
            # 기존 한국어 환경 및 의존성 라이브러리와 100% 호환되는 경량 한국어 문장 임베딩 모델 기용
            model_name = "jhgan/ko-sroberta-multitask"
            _math_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _math_model = AutoModel.from_pretrained(model_name)
        
        # 텍스트 토큰화 및 텐서 변환
        inputs_sim = _math_tokenizer(summary_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs_orig = _math_tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs_sim = _math_model(**inputs_sim)
            outputs_orig = _math_model(**inputs_orig)
            
        # Mean Pooling 기법을 통해 문장의 핵심 문맥 벡터 추출
        emb_sim = outputs_sim.last_hidden_state.mean(dim=1)
        emb_orig = outputs_orig.last_hidden_state.mean(dim=1)
        
        # 기하학적 코사인 유사도 수학 공식 다이렉트 연산
        cosine_sim = torch.nn.functional.cosine_similarity(emb_sim, emb_orig).item()
        
        # 유사도 결과 범위(-1 ~ 1)를 점수 표준 범위(0 ~ 100)로 변환 및 보정
        math_score = max(0.0, min(1.0, cosine_sim)) * 100.0
        return round(math_score, 1)
    except Exception as e:
        print(f"(수학적 유사도 에러: {e}) ", end="")
        return None

def evaluate_models():
    print("\n" + "="*50)
    print("🎯 [AI 심사위원 평가 + MLOps 메트릭 로깅 시작]")
    
    response = supabase.table("news_data").select("*").is_("score_kobart", "null").order("id", desc=True).execute()
    

    # 수정 코드 (조건을 지워서 이미 점수가 있는 과거 데이터까지 전부 수집)
    ## response = supabase.table("news_data").select("*").order("id", desc=True).execute()


    if not response.data:
        print("💡 평가할 새로운 데이터가 없습니다.")
        return

    # 🚀 MLflow 초기화 및 룸 세팅
    target_run_name = "Continuous_MLOps_Monitoring"
    active_run_id = None
    run_context = None
    
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("News_Summary_MLOps_Project")
        
        current_experiment = mlflow.get_experiment_by_name("News_Summary_MLOps_Project")
        if current_experiment:
            existing_runs = mlflow.search_runs(
                experiment_ids=[current_experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{target_run_name}'",
                max_results=1
            )
            if not existing_runs.empty:
                active_run_id = existing_runs.iloc[0]["run_id"]
                print(f"🔗 [MLflow 연결] 기존 모니터링 룸(Run ID: {active_run_id})을 재사용하여 그래프를 이어 그립니다.")
                
        run_context = mlflow.start_run(run_id=active_run_id, run_name=target_run_name)
    except Exception as e:
        print(f"⚠️ [MLflow Warning] 트래킹 서버 미가동(또는 클라우드 환경). 대시보드 전송을 우회하고 DB 적재에 집중합니다.")

    # 실질적인 평가 및 DB 적재 루프 진입
    try:
        for news in response.data:
            news_id = news['id']
            topic = news.get('topic', '미분류')
            print(f"\n📝 대상 기사 ID: {news_id} [{topic}]")
            
            update_data = {}
            for m in ["kobart", "kot5", "roberta"]:
                summary = news.get(f"summary_{m}")
                if not summary or len(summary) < 5: continue
                
                print(f" > {m.upper()} 하이브리드 종합 채점 시작...", flush=True)
                
                # 1단계: 4대 다각화 지표 병렬 수집
                s1 = get_gpt_score(summary, news['description'])   # 정성평가 1
                s2 = get_clova_score(summary, news['description']) # 정성평가 2
                s3 = get_comprehensive_score(summary, news['description']) # 팀원의 학술지표 (QAFactEval & G-Eval)
                s4 = get_math_similarity_score(summary, news['description']) # 100% 수학적 객관적 지표
                
                # 2단계: 정성/학술 지표 그룹 앙상블 평균 계산
                llm_academic_scores = [s for s in [s1, s2, s3] if s is not None]
                
                # 3단계: 하이브리드 가중치 결합 알고리즘 가동
                if llm_academic_scores and s4 is not None:
                    avg_llm_academic = sum(llm_academic_scores) / len(llm_academic_scores)
                    # [설계 핵심] 주관성을 흐리기 위해 정성/학술 지표 70% + 절대적 수학적 유사도 지표 30% 반영
                    final_score = round((avg_llm_academic * 0.7) + (s4 * 0.3), 1)
                elif llm_academic_scores:
                    final_score = round(sum(llm_academic_scores) / len(llm_academic_scores), 1)
                elif s4 is not None:
                    final_score = s4
                else:
                    final_score = 0.0
                
                # 4단계: 가드레일 클리핑 보정
                if final_score > 0:
                    if final_score > 100.0: final_score = 100.0
                    elif final_score < 0.0: final_score = 0.0
                        
                    update_data[f"score_{m}"] = final_score
                    print(f"   └─> [매트릭스 확인] GPT: {s1} | Clova: {s2} | 학술: {s3} | 수학적유사도: {s4}")
                    print(f"   └─> 🏆 최종 가중치 결합 점수: {final_score}점")
                    
                    # MLflow 로깅 세션이 안전하게 수립되었을 때만 메트릭 전송 시도
                    if run_context and mlflow.active_run():
                        try:
                            mlflow.log_metric(f"score_{m}_{topic}", final_score, step=int(news_id))
                        except Exception:
                            pass
                else:
                    update_data[f"score_{m}"] = 0.0
                    print("   └─> ❌ 평가 불가 (0점 처리)")
                time.sleep(0.5)

            if update_data:
                supabase.table("news_data").update(update_data).eq("id", news_id).execute()

        print("\n🏆 DB 업데이트 및 인프라 파이프라인 정상 종료!")
        
    finally:
        if run_context:
            try:
                mlflow.end_run()
            except Exception:
                pass

if __name__ == "__main__":
    evaluate_models()