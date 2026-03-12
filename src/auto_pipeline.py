import os
import time
from dotenv import load_dotenv
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from preprocess import clean_text

# 1. DB 접속
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 2. AI 요약 세팅
print("AI 요약 가동 준비 중")
model_name = "digit82/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 요약 안 된 뉴스 가져오기
print("DB에서 새 뉴스를 확인")
response = supabase.table("news_data").select("*").is_("summary", "null").execute()
news_list = response.data

if not news_list:
    print("현재 요약할 새로운 뉴스가 없습니다.")
    exit()

print(f"총 {len(news_list)}건의 요약 작업 대기열을 발견했습니다.\n")

# 4. 스마트 요약 파이프라인
for news in news_list:
    news_id = news['id'] 
    
    # 본문(description)을 먼저 찾고, 없으면 제목(title)을 씁니다.
    raw_text = news.get('description', '') 
    if not raw_text or len(raw_text) < 10: 
        raw_text = news.get('title', '') 
        
    if not raw_text:
        continue 

    print(f"처리 중: [{news['title']}]")
    clean_news = clean_text(raw_text)
    
    # ★ 실무형 안전장치: 텍스트가 너무 짧으면 요약 모델을 돌리지 않고 그대로 사용합니다.
    if len(clean_news) < 40:
        print("텍스트가 너무 짧아 원문을 그대로 사용")
        ai_summary = clean_news
    else:
        # 텍스트가 충분히 길 때만 AI 요약 가동
        inputs = tokenizer(clean_news, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=100,
            min_length=15, # 최소 길이를 15로 줄여서 반복을 방지합니다.
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        ai_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # [DB 업데이트 (저장)]
    supabase.table("news_data").update({"summary": ai_summary}).eq("id", news_id).execute()
    
    print(f"완료: {ai_summary}\n")
    time.sleep(1)

print("파이프라인 가동이 완벽하게 끝났습니다.")