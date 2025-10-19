# PDF 파일을 Pinecone에 임베딩하는 완전 가이드

## 📋 개요

PDF 파일은 DOCX와 달리 일반 텍스트 추출이 어렵습니다. 이 가이드는 GPT-4의 비전 능력을 활용하여 PDF를 마크다운으로 변환한 후, Pinecone 벡터 데이터베이스에 저장하는 전체 과정을 설명합니다.

---

## 참고 자료
https://github.com/jasonkang14/inflearn-agent-use-cases-lecture/blob/main/14.1%20%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC.ipynb

---
## 🔧 필수 라이브러리 설치

```bash
pip install pyzerox nest-asyncio langchain-text-splitters langchain-pinecone langchain-openai python-dotenv
```

---

## 1️⃣ 단계 1: 비동기 처리 설정

```python
import nest_asyncio

# 이미 실행 중인 event loop가 있을 때 비동기 함수 실행 가능하게 함
nest_asyncio.apply()
```

**목적**: Jupyter Notebook이나 IPython 환경에서 여러 PDF를 동시에 빠르게 처리하기 위함

---

## 2️⃣ 단계 2: PDF를 마크다운으로 변환

```python
from pyzerox import zerox
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# 사용할 OpenAI 모델 (GPT-4의 비전 능력 필요)
model = "gpt-4o-2024-11-20"

async def pdf_to_markdown():
    """
    PDF 파일을 마크다운으로 변환하는 함수
    """
    input_dir = "./documents"  # PDF 파일이 있는 폴더
    output_dir = "./output"    # 변환된 마크다운을 저장할 폴더
    
    # output 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 폴더 내 모든 PDF 파일 처리
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            file_path = f"{input_dir}/{filename}"
            
            print(f"⏳ 변환 중: {filename}")
            
            # PDF를 마크다운으로 변환
            result = await zerox(
                file_path=file_path,
                model=model,
                output_dir=output_dir
            )
            
            print(f"✅ 완료: {filename}")
    
    return result

# 실행
result = asyncio.run(pdf_to_markdown())
```

**결과**: `./output/` 폴더에 마크다운 파일 생성

---

## 3️⃣ 단계 3: 마크다운을 계층별로 분할

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운의 헤더 구조를 기반으로 분할
headers_to_split_on = [
    ("#", "title"),       # 1단계 제목
    ("##", "chapter"),    # 2단계 제목
    ("###", "section"),   # 3단계 제목
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
```

**설명**: 마크다운의 헤더 구조를 인식하여 의미 있는 단위로 텍스트를 분할합니다.

---

## 4️⃣ 단계 4: Pinecone 벡터 스토어 설정

```python
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Pinecone 인덱스 이름
index_name = "pdf-documents-index"

# 벡터 스토어 초기화
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)
```

**참고**: 환경변수에 `OPENAI_API_KEY`와 `PINECONE_API_KEY`를 설정해야 합니다.

---

## 5️⃣ 단계 5: 마크다운 파일을 읽고 Pinecone에 저장 (실제 운영 코드)

```python
import os

input_dir = './output'  # 변환된 마크다운 파일들이 있는 폴더

print("🚀 Pinecone에 문서 저장 시작...\n")

# output 폴더의 모든 마크다운 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.md'):
        md_path = os.path.join(input_dir, filename)
        
        print(f"📄 처리 중: {filename}")
        
        # 마크다운 파일 읽기
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # 마크다운 분할
        docs = markdown_splitter.split_text(markdown_text)
        
        # 각 document에 source 메타데이터 추가
        for doc in docs:
            doc.metadata['source'] = filename.replace('.md', '')
        
        # Pinecone에 한번에 저장
        vector_store.add_documents(docs)
        
        print(f"✅ 저장 완료: {len(docs)}개 chunks\n")

print("🎉 모든 문서 저장 완료!")
```

**핵심 포인트**:
- ✅ **모든 파일** 처리 (break문 제거)
- ✅ **모든 chunk** 저장 (내부 break문 제거)
- ✅ source 메타데이터 추가로 어느 문서에서 왔는지 추적 가능

---

## 6️⃣ 단계 6: 저장된 데이터 검색

```python
# 리트리버 생성
retriever = vector_store.as_retriever()

# 검색 실행
query = "원 복리후생 및 복지 FAQ"
results = retriever.invoke(query)

# 결과 출력
print(f"🔍 '{query}'에 대한 검색 결과:\n")
for i, doc in enumerate(results, 1):
    print(f"[결과 {i}]")
    print(f"출처: {doc.metadata.get('source', '미지정')}")
    print(f"제목: {doc.metadata.get('title', '미지정')}")
    print(f"내용: {doc.page_content[:200]}...")
    print("-" * 50)
```

---

## 📊 전체 워크플로우 요약

```
PDF 파일들
   ↓
[단계 2] pyzerox + GPT-4로 변환
   ↓
마크다운 파일들 (./output/)
   ↓
[단계 3] MarkdownHeaderTextSplitter로 분할
   ↓
Document chunks (헤더 정보 포함)
   ↓
[단계 4~5] OpenAIEmbeddings로 벡터화
   ↓
Pinecone 벡터 데이터베이스에 저장
   ↓
[단계 6] 검색 및 활용
```

---

## 🔑 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **nest_asyncio** | 여러 PDF를 동시에 빠르게 처리 |
| **pyzerox** | GPT-4 비전으로 PDF를 마크다운으로 변환 |
| **MarkdownHeaderTextSplitter** | 마크다운 헤더 구조를 인식하여 의미 있게 분할 |
| **OpenAIEmbeddings** | 텍스트를 벡터(숫자)로 변환 |
| **PineconeVectorStore** | 벡터를 저장하고 유사도 검색 수행 |
| **메타데이터** | 문서의 출처, 제목 등 추가 정보 저장 |

---

## ⚠️ 주의사항

1. **API 비용**: GPT-4 Vision API 사용 (PDF당 비용 발생)
2. **환경변수 설정**:
   ```
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   ```
3. **폴더 구조**:
   ```
   project/
   ├── documents/          # PDF 파일 넣기
   ├── output/            # 변환된 마크다운 저장
   └── your_script.py
   ```

---

## 💡 추가 팁

- 마크다운 헤더가 없는 PDF는 pyzerox가 자동으로 구조화
- 검색 결과 개수 조절: `retriever.invoke(query, k=5)` (상위 5개)
- 메타데이터 필터링: `retriever.invoke(query, filter={"source": "특정문서"})`