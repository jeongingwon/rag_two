# Pinecone 개념 정리

## 1. Pinecone이란?
**벡터 데이터베이스(Vector Database)** 서비스. 텍스트/이미지 등을 임베딩(embedding) 벡터로 변환한 뒤 저장하고, 의미적으로 유사한 벡터를 빠르게 검색할 수 있는 완전관리형(fully managed) 클라우드 서비스다.

## 2. 왜 필요한가 (RAG에서의 역할)
일반 DB는 키워드/정확한 값으로 검색하지만, RAG(Retrieval-Augmented Generation)는 "의미가 비슷한 문서"를 찾아야 한다.

```
문서 → 임베딩 모델 → 벡터 → Pinecone에 저장(upsert)
질문 → 임베딩 모델 → 벡터 → Pinecone에서 유사 벡터 검색(query)
검색된 문서 → LLM에 context로 전달 → 답변 생성
```

## 3. 핵심 개념

| 개념 | 설명 |
|---|---|
| Index | 벡터를 저장하는 컨테이너 (RDB의 테이블과 유사) |
| Vector | 임베딩된 숫자 배열 (예: 1536차원) |
| Dimension | 벡터의 차원 수. 임베딩 모델과 반드시 일치해야 함 |
| Metric | 유사도 계산 방식 (`cosine`, `euclidean`, `dotproduct` 등) |
| Upsert | Update + Insert. 벡터 삽입/갱신 (id 중복 시 덮어씀) |
| Query | 질문 벡터와 가장 가까운 top-k개 벡터 검색 |
| Metadata | 벡터와 함께 저장하는 원본 텍스트/속성 (필터링·표시용) |
| ServerlessSpec | 서버리스 인덱스의 클라우드/리전 설정 (예: AWS `us-east-1`) |

## 4. 사용 흐름 (ch11.ipynb 기준)

### 4-1. 설치 및 초기화
```python
# !uv add pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
```

### 4-2. 인덱스 생성 (없을 때만)
```python
index_name = "quickstart-index"

if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,          # text-embedding-3-small 기준
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
```

### 4-3. 문서 임베딩 후 저장 (Upsert)
```python
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=[doc["text"] for doc in documents],
)

vectors = [
    {"id": doc["id"], "values": emb.embedding, "metadata": {"text": doc["text"]}}
    for doc, emb in zip(documents, response.data)
]

index.upsert(vectors=vectors)
```

### 4-4. 질문 벡터화 후 검색 (Query)
```python
question_vector = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=question,
).data[0].embedding

result = index.query(
    vector=question_vector,
    top_k=2,
    include_metadata=True,
)

for match in result.matches:
    print(match.id, match.metadata["text"], match.score)
```

## 5. 요약
- Pinecone = **임베딩 벡터 저장소 + 유사도 검색 엔진**
- OpenAI 임베딩(또는 다른 임베딩 모델)과 짝을 이뤄 RAG 파이프라인의 "검색(Retrieval)" 단계를 담당
- 핵심 API: `create_index` → `upsert` → `query`
- 인덱스의 `dimension`은 사용하는 임베딩 모델의 출력 차원과 반드시 일치해야 함 (`text-embedding-3-small` = 1536차원)

## 6. 보안 주의사항
- `PINECONE_API_KEY`, `OPENAI_API_KEY` 등은 `.env` 파일로 관리하고 `.gitignore`에 포함시킬 것
- 노트북 실행 결과(output) 셀에 API 키 등 민감 정보가 `print()` 등으로 출력된 채 저장되지 않도록 커밋 전 확인 필요
