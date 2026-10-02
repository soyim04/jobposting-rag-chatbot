"""평가용 청크를 임베딩해 ChromaDB에 저장하는 색인 스크립트 (4단계).

data/chunks/eval/strategy_{a,b}.jsonl을 읽어 전략별 컬렉션에 저장한다.
메타데이터는 docs/step3_metadata.md의 "벡터 DB 저장 형식"대로 변환한다.
- 리스트(job_categories, tech_stacks)는 쉼표로 이은 문자열 (설계 원칙 3: 벡터 DB를 바꿔도 같은 형식)
- closed_at은 문자열과 함께 정수(closed_at_int, 예: 20261014)로도 저장 (Chroma는 문자열 크기 비교 불가, 1.5.9에서 확인)
- status는 수집 시점 값이라 넣지 않는다 (마감 여부는 posting_status.effective_status로 계산)

A·B에서 본문이 같은 청크는 한 번만 임베딩한다. 사용 토큰과 비용은 eval/usage_log.csv에 쌓는다.

사용법
    .venv\\Scripts\\python.exe src\\build_index.py
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks" / "eval"
CHROMA_DIR = ROOT / "chroma_db"
USAGE_LOG = ROOT / "eval" / "usage_log.csv"

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_1M = 0.02  # 달러, OpenAI 가격표 2026-10-02 확인
BATCH_SIZE = 100
STRATEGIES = {"A": "strategy_a.jsonl", "B": "strategy_b.jsonl"}
LIST_FIELDS = ("job_categories", "tech_stacks")
DROP_FIELDS = ("text", "status")


def collection_name(strategy):
    return f"eval-{strategy.lower()}-{EMBED_MODEL}"


def to_metadata(chunk):
    meta = {k: v for k, v in chunk.items() if k not in DROP_FIELDS}
    for field in LIST_FIELDS:
        meta[field] = ", ".join(meta[field])
    meta["closed_at_int"] = int(chunk["closed_at"].replace("-", "")) if chunk["closed_at"] != "상시" else 99991231
    return meta


def embed_all(client, texts):
    vectors, total_tokens = [], 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in sorted(res.data, key=lambda d: d.index))
        total_tokens += res.usage.total_tokens
        print(f"  임베딩 {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors, total_tokens


def log_usage(row):
    is_new = not USAGE_LOG.exists()
    with USAGE_LOG.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    load_dotenv(ROOT / ".env")
    chunks = {s: [json.loads(line) for line in (CHUNKS_DIR / f).open(encoding="utf-8")]
              for s, f in STRATEGIES.items()}
    unique_texts = sorted({c["text"] for cs in chunks.values() for c in cs})
    print(f"청크 A {len(chunks['A'])}개, B {len(chunks['B'])}개 → 고유 본문 {len(unique_texts)}개 임베딩")

    vectors, tokens = embed_all(OpenAI(), unique_texts)
    vector_of = dict(zip(unique_texts, vectors))

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for strategy, cs in chunks.items():
        name = collection_name(strategy)
        if name in [c.name for c in db.list_collections()]:
            db.delete_collection(name)
        col = db.create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[c["chunk_id"] for c in cs],
            documents=[c["text"] for c in cs],
            embeddings=[vector_of[c["text"]] for c in cs],
            metadatas=[to_metadata(c) for c in cs],
        )
        print(f"  컬렉션 {name}: {col.count()}개 저장")

    cost = tokens / 1_000_000 * EMBED_PRICE_PER_1M
    log_usage({
        "시각": datetime.now().isoformat(timespec="seconds"),
        "작업": "build_index",
        "모델": EMBED_MODEL,
        "입력 토큰": tokens,
        "출력 토큰": 0,
        "비용(달러)": f"{cost:.5f}",
        "메모": f"고유 본문 {len(unique_texts)}개, 컬렉션 A {len(chunks['A'])} / B {len(chunks['B'])}",
    })
    print(f"임베딩 토큰 {tokens:,}개, 비용 약 ${cost:.4f} → {USAGE_LOG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
