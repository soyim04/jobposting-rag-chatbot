"""top_k별 검색 적중률 비교 (6단계, LLM 호출 없이 질문 임베딩만 사용).

정답 근거가 있는 25문항에 대해 질문을 임베딩해 전략 B 컬렉션에서 top-k를 가져오고,
정답 근거 청크(공고ID/항목)가 하나라도 들어 있으면 적중, 들어 있는 비율을 근거 재현율로 센다.
- 질문 그대로 (1차 기준선 설정)
- 동의어 확장 (2단계 설정, data/tech_synonyms.json)

결과: eval/results/topk_retrieval.csv (비용은 임베딩 약 $0.00002)

사용법
    .venv\\Scripts\\python.exe src\\topk_retrieval.py
"""

import csv

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

import run_eval as r
from build_index import EMBED_MODEL, EMBED_PRICE_PER_1M, collection_name, log_usage
from datetime import datetime

KS = [3, 5, 10, 20]
OUT = r.ROOT / "eval" / "results" / "topk_retrieval.csv"


def main():
    load_dotenv(r.ROOT / ".env")
    client = OpenAI()
    items = [it for it in r.load_csv(r.EVAL_SET) if r.gold_units(it["근거"])]  # 정답 근거가 있는 25문항
    groups = r.load_synonym_groups()
    col = chromadb.PersistentClient(path=str(r.CHROMA_DIR)).get_collection(collection_name("B"))
    rows, tokens = [], 0
    for mode in ("질문 그대로", "동의어 확장"):
        qs = [it["질문"] if mode == "질문 그대로" else r.expand_query(it["질문"], groups)[0] for it in items]
        emb = client.embeddings.create(model=EMBED_MODEL, input=qs)
        tokens += emb.usage.total_tokens
        qvecs = [d.embedding for d in sorted(emb.data, key=lambda d: d.index)]
        for k in KS:
            hits, recalls = 0, []
            for it, qv in zip(items, qvecs):
                ids = r.scope_ids(it["검색 범위"])
                where = {"posting_id": {"$in": ids}} if ids else None
                res = col.query(query_embeddings=[qv], n_results=k, where=where)
                got = [(m["posting_id"], m["section_key"]) for m in res["metadatas"][0]]
                gold = r.gold_units(it["근거"])
                found = [g for g in gold if g in got]
                hits += bool(found)
                recalls.append(len(found) / len(gold))
            rows.append({"설정": mode, "top_k": k, "적중": f"{hits}/{len(items)}",
                         "검색 적중률(%)": round(hits / len(items) * 100, 1),
                         "근거 재현율(%)": round(sum(recalls) / len(recalls) * 100, 1)})
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cost = tokens / 1e6 * EMBED_PRICE_PER_1M
    log_usage({"시각": datetime.now().isoformat(timespec="seconds"), "작업": "topk_retrieval", "모델": EMBED_MODEL,
               "입력 토큰": tokens, "출력 토큰": 0, "비용(달러)": f"{cost:.5f}", "메모": "top_k별 검색 적중률, 25문항×2설정"})
    for row in rows:
        print(row)
    print(f"→ {OUT.relative_to(r.ROOT).as_posix()}, 비용 약 ${cost:.6f}")


if __name__ == "__main__":
    main()
