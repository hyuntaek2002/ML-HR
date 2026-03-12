import re

def clean_text(text):
    
    # 1. 이메일 주소 제거
    text = re.sub(r'[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text)
    # 2. HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 3. 특수문자 제거 (한글, 영문, 숫자, 기본 구두점 빼고 다 날리기)
    text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', ' ', text)
    # 4. 다중 공백을 하나의 공백으로 압축
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text