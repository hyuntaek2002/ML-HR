import os
import time
import torch
import traceback
import re
from dotenv import load_dotenv
from supabase import create_client
# 💡 AutoTokenizer 외에 T5 전용 명시적 로더인 T5Tokenizer를 추가 임포트함
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from preprocess import clean_text

# 1. 환경 설정
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 사용자님이 원래 지정하셨던 오리지널 모델 라인업 100% 유지
MODEL_CONFIGS = {
    "kobart": "digit82/kobart-summarization",
    "kot5": "paust/pko-t5-base",
    "roberta": "EbanLee/kobart-summary-v3"
}

def run_summarization():
    """DB에서 요약 대기 데이터를 가져와 모델별 일괄 로드 패턴으로 요약을 진행합니다."""
    print("🔍 DB에서 요약 대기 데이터(전 분야) 확인 중...")
    
    # 요약이 하나라도 비어있는(null) 뉴스 호출
    response = supabase.table("news_data").select("*").is_("summary_kobart", "null").execute()
    news_list = response.data

    if not news_list:
        print("💡 현재 요약할 새로운 뉴스가 없습니다.")
        return

    print(f"📝 총 {len(news_list)}건의 뉴스 요약을 시작합니다. (GitHub Actions 메모리 최적화 모드)")

    # 1. 텍스트 전처리 미리 세팅
    cleaned_news_list = []
    for news in news_list:
        raw_text = news.get('description') or news.get('title') or ""
        clean_news = clean_text(re.sub('<[^>]*>', '', raw_text))
        cleaned_news_list.append({
            "id": news['id'],
            "title": news['title'],
            "topic": news.get('topic', '미분류'),
            "clean_text": clean_news,
            "summaries": {}  # 각 모델별 결과가 여기에 담김
        })

    # 2. 🚀 [MLOps 핵심 최적화] 모델별로 루프를 돌려 하드디스크 로드 횟수를 3회로 제한
    for model_name, model_id in MODEL_CONFIGS.items():
        print(f"\n🤖 [{model_name}] 글로벌 모델 가중치 로드 중... ({model_id})")
        
        try:
            # 💡 [치명적 결함 해결] kot5(paust) 모델은 최신 AutoTokenizer와 충돌하므로,
            # T5Tokenizer를 직접 명시하여 불러오면 레거시 파일 에러를 완벽하게 우회함
            if model_name == "kot5":
                tokenizer = T5Tokenizer.from_pretrained(
                    model_id, 
                    token=os.getenv("HF_TOKEN")
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    token=os.getenv("HF_TOKEN"),
                    use_fast=True
                )
                
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # 로드된 하나의 모델로 모든 뉴스를 긁어 밀어 넣기
            for item in cleaned_news_list:
                print(f"   └> ID {item['id']} ({item['topic']}) 요약문 추론 중...")
                input_text = f"summarize: {item['clean_text']}" if model_name == "kot5" else item['clean_text']
                
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
                item["summaries"][model_name] = summary

            # 사용 완료된 모델 즉시 가비지 컬렉션 (RAM 확보)
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ⚠️ [{model_name}] 파이프라인 구동 중 에러 발생: {e}")
            for item in cleaned_news_list:
                item["summaries"][model_name] = "요약 실패 (에러 확인 필요)"

    # 3. 💾 최종 완료된 3개 모델 결과를 모아서 Supabase에 한 번에 적재
    print("\n💾 요약 완료 데이터 Supabase 적재 시작...")
    for item in cleaned_news_list:
        try:
            supabase.table("news_data").update({
                "summary_kobart": item["summaries"].get("kobart", "요약 실패"),
                "summary_kot5": item["summaries"].get("kot5", "요약 실패"),
                "summary_roberta": item["summaries"].get("roberta", "요약 실패")
            }).eq("id", item['id']).execute()
            print(f"   ✅ ID {item['id']} ({item['topic']}) 최종 업데이트 완료")
            time.sleep(0.2)
        except Exception as e:
            print(f"   ❌ ID {item['id']} DB 적재 실패: {e}")

if __name__ == "__main__":
    run_summarization()