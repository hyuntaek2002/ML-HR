import os
import re
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from supabase import create_client
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModel
)
from preprocess import clean_text

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

models = {
    "KoBART": "digit82/kobart-summarization",
    "KoT5": "lcw99/t5-base-korean-text-summary",
    "KLUE-RoBERTa": "klue/roberta-base"
}

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 처리할 뉴스 개수를 8개로 제한
    response = supabase.table("news_data").select("*").is_("summary_kobart", "null").limit(8).execute()
    
    if not response.data:
        print("새로 요약할 뉴스가 없습니다.")
        return

    print(f"총 {len(response.data)}건의 뉴스 요약을 시작합니다. (장치: {device})")

    for news in response.data:
        print(f"\n기사: {news['title'][:30]}...")

        description = news.get('description')
        if not description or len(str(description).strip()) < 20:
            print("(본문이 너무 짧거나 비어있어 요약을 건너뜁니다.")
            supabase.table("news_data").update({
                "summary_kobart": "본문 부족",
                "summary_kot5": "본문 부족",
                "summary_roberta": "본문 부족"
            }).eq("id", news["id"]).execute()
            continue

        raw_text = clean_text(str(description))
        update_data = {}

        for m_type, m_name in models.items():
            try:
                if m_type == "KoBART":
                    tok = AutoTokenizer.from_pretrained(m_name, use_fast=False)
                    mod = AutoModelForSeq2SeqLM.from_pretrained(m_name).to(device)
                    inputs = tok(raw_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    out = mod.generate(inputs["input_ids"], num_beams=4, max_length=128)
                    update_data["summary_kobart"] = tok.decode(out[0], skip_special_tokens=True)

                elif m_type == "KoT5":
                    tok = AutoTokenizer.from_pretrained(m_name, use_fast=False)
                    mod = AutoModelForSeq2SeqLM.from_pretrained(m_name).to(device)
                    inputs = tok("summarize: " + raw_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    out = mod.generate(inputs["input_ids"], max_length=128)
                    update_data["summary_kot5"] = tok.decode(out[0], skip_special_tokens=True)

                elif m_type == "KLUE-RoBERTa":
                    tok = AutoTokenizer.from_pretrained(m_name)
                    mod = AutoModel.from_pretrained(m_name).to(device)
                    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if len(s.strip()) > 5]
                    
                    if len(sents) < 2:
                        update_data["summary_roberta"] = raw_text[:100]
                    else:
                        doc_inputs = tok(raw_text, return_tensors="pt", truncation=True).to(device)
                        with torch.no_grad(): doc_emb = mod(**doc_inputs).last_hidden_state[:, 0, :]
                        sent_inputs = tok(sents, padding=True, truncation=True, return_tensors="pt").to(device)
                        with torch.no_grad(): sent_embs = mod(**sent_inputs).last_hidden_state[:, 0, :]
                        sims = F.cosine_similarity(sent_embs, doc_emb)
                        update_data["summary_roberta"] = sents[sims.argmax().item()]

                del mod, tok
                if device == "cuda": torch.cuda.empty_cache()

            except Exception as e:
                # 에러 발생 시 데이터베이스 컬럼명과 정확히 일치하도록 예외 처리 수정
                if m_type == "KoBART":
                    update_data["summary_kobart"] = f"Error"
                elif m_type == "KoT5":
                    update_data["summary_kot5"] = f"Error"
                elif m_type == "KLUE-RoBERTa":
                    update_data["summary_roberta"] = f"Error"

        supabase.table("news_data").update(update_data).eq("id", news["id"]).execute()
        print("3개 모델 요약 완료 및 DB 저장")

if __name__ == "__main__":
    run_evaluation()
