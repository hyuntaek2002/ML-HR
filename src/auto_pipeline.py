import os
import time
import torch
import traceback
import re
from dotenv import load_dotenv
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from preprocess import clean_text

# 1. 환경 설정
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 비교할 3가지 모델 정의 (오리지널 라인업 유지)
MODEL_CONFIGS = {
    "kobart": "digit82/kobart-summarization",
    "kot5": "paust/pko-t5-base",
    "roberta": "EbanLee/kobart-summary-v3"
}

import mlflow

@mlflow.trace(name="generate_with_model", span_type="LLM")
def generate_with_model(model_name, text):
    """지정된 모델을 사용하여 요약문을 생성합니다."""
    model_id = MODEL_CONFIGS[model_name]
    print(f"   > [{model_name}] 모델 추론 시작...")
    
    # 가비지 컬렉션을 위해 변수 초기화
    model = None
    tokenizer = None
    
    try:
        # T5 계열은 'summarize: ' 접두사 사용
        input_text = f"summarize: {text}" if model_name == "kot5" else text
        
        # [중요] 사용자 검증 완료 오리지널 치트키 옵션 (use_fast=False) 그대로 복구
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=os.getenv("HF_TOKEN"),
            use_fast=False 
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # 토크나이징 및 생성
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=150,
            min_length=20,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print(f"   ⚠️ [{model_name}] 에러 발생: {e}")
        return f"요약 실패 (에러 확인 필요)"

    finally:
        # 에러 발생 여부와 무관하게 추론 종료 시 즉시 가비지 컬렉션 가동 (GitHub Actions OOM 방어)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_summarization():
    """DB에서 요약이 필요한 데이터를 찾아 3개 모델로 요약을 진행합니다."""
    print("🔍 DB에서 요약 대기 데이터(전 분야) 확인 중...")
    
    # 요약이 하나라도 비어있는(null) 뉴스 호출
    response = supabase.table("news_data").select("*").is_("summary_kobart", "null").execute()
    news_list = response.data

    if not news_list:
        print("💡 현재 요약할 새로운 뉴스가 없습니다.")
        return

    print(f"📝 총 {len(news_list)}건의 뉴스 요약을 시작합니다.")

    for news in news_list:
        news_id = news['id']
        news_topic = news.get('topic', '미분류') # 어떤 분야인지 파악
        raw_text = news.get('description') or news.get('title')
        
        # HTML 태그 제거 및 텍스트 클리닝
        clean_news = clean_text(re.sub('<[^>]*>', '', raw_text))
        
        print(f"\n[작업 시작] ID: {news_id} | 분야: {news_topic} | 제목: {news['title'][:20]}...")
        
        # 3개 모델 순차적 요약 (오리지널 함수형 구조 복구)
        sum_kobart = generate_with_model("kobart", clean_news)
        sum_kot5 = generate_with_model("kot5", clean_news)
        sum_roberta = generate_with_model("roberta", clean_news)

        # 결과 업데이트
        supabase.table("news_data").update({
            "summary_kobart": sum_kobart,
            "summary_kot5": sum_kot5,
            "summary_roberta": sum_roberta
        }).eq("id", news_id).execute()
        
        print(f"✅ ID {news_id} ({news_topic}) 저장 완료")
        time.sleep(0.5) # DB 과부하 방지용 짧은 휴식

if __name__ == "__main__":
    run_summarization()