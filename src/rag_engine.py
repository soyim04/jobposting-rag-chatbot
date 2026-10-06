"""질문 1개에 답하는 서비스용 엔진 (7단계).

6단계까지 실험으로 정한 파이프라인을 질문 하나씩 처리하도록 묶었다. 평가용 코드(run_eval.py)는 건드리지 않고,
그 안의 함수들을 가져다 쓰되 데이터(공고 표, 공고별 칸, 컬렉션)와 기준일만 갈아 끼운다.

처리 흐름 (docs/step6_experiment_log.md)
1. 공고를 골랐으면 그 공고 전체를 근거로 넣는다 (1단계)
2. 아니면 질문을 조건(학력·재택·마감일·기술 등)으로 해석해 코드로 거른다 (3단계). 조건이 없으면 4로
3. 질문의 기술 표기에 동의어를 덧붙여 검색한다 (2단계)
4. 회사 2곳 이상 신입 비교나 조건 없는 필터 질문이면 공고마다 그 항목 청크를 가져오고 (6단계),
   그 밖에는 질문과 가까운 청크 top-k개를 가져온다 (top-k 기본 10)
5. 점핏 분류와 본문이 다른 공고는 그 차이를 근거에 붙이고 (4단계), 답변 지시문 v2로 답한다 (5단계)

주의
- run_eval.py의 모듈 전역값(POSTINGS, REF_DATE, FIELDS_CSV, 지시문 안의 날짜)을 바꿔서 쓰므로, 한 프로세스에서는 한 번에 하나의
  데이터셋만 쓸 수 있다.
- 공고 원문은 재배포하지 않는다. 화면에는 출처(회사·공고명·항목·마감일·원문 링크)만 돌려주고 인용 문장은 로그용으로만 둔다.

사용법
    .venv\\Scripts\\python.exe src\\rag_engine.py --dataset eval --scope "필터: 신입" "재택이나 유연근무 되는 곳 있어?"
"""

import argparse
import time
from datetime import date

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

import run_eval as r
from build_index import EMBED_MODEL, EMBED_PRICE_PER_1M
from posting_status import effective_status

ROOT = r.ROOT
ORIG_REF_DATE = r.REF_DATE  # 지시문 문자열에 박혀 있는 평가 기준일 (바꿔 끼울 때 이 값을 찾아 교체한다)
ORIG_PARSE_SYSTEM = r.PARSE_SYSTEM

DATASETS = {
    # 평가용: 9/30 스냅샷, 기준일 고정
    "eval": {"postings": r.SNAPSHOT, "fields": ROOT / "data" / "posting_fields_eval.csv",
             "collection": f"eval-b-{EMBED_MODEL}", "ref_date": ORIG_REF_DATE},
    # 서비스용: 최신 재수집본, 기준일은 오늘
    "service": {"postings": ROOT / "data" / "postings.csv", "fields": ROOT / "data" / "posting_fields_service.csv",
                "collection": f"service-b-{EMBED_MODEL}", "ref_date": None},
}
DEFAULTS = {"model": "gpt-6-luna", "top_k": 10, "per_posting": 2, "prompt": "v2", "reasoning": None}


class Engine:
    def __init__(self, dataset="service", ref_date=None, **options):
        load_dotenv(ROOT / ".env")
        self.client = OpenAI()
        self.opt = {**DEFAULTS, **options}
        cfg = DATASETS[dataset]
        self.dataset = dataset
        self.ref_date = ref_date or cfg["ref_date"] or date.today().isoformat()
        # run_eval의 전역값을 이 데이터셋으로 바꾼다
        r.POSTINGS = {p["posting_id"]: p for p in r.load_csv(cfg["postings"]) if p["included"] == "Y"}
        r.FIELDS_CSV = cfg["fields"]
        r.REF_DATE = self.ref_date
        r.PARSE_SYSTEM = ORIG_PARSE_SYSTEM.replace(ORIG_REF_DATE, self.ref_date)
        self.postings = r.POSTINGS
        self.col = chromadb.PersistentClient(path=str(r.CHROMA_DIR)).get_collection(cfg["collection"])
        self.groups = r.load_synonym_groups()
        base, structured, diff = r.PROMPTS[self.opt["prompt"]]
        self.rules = tuple(x.replace(ORIG_REF_DATE, self.ref_date) for x in (base, structured, diff))

    # ---- 화면용 도우미 -------------------------------------------------
    def status_of(self, pid):
        p = self.postings[pid]
        return effective_status(p["closed_at"], p["status"], self.ref_date)

    def posting_options(self):
        """공고 선택 목록: 진행 중인 공고를 앞에, 마감 공고를 뒤에 둔다."""
        rows = [{"posting_id": pid, "company": p["company"], "title": p["title"], "closed_at": p["closed_at"],
                 "status": self.status_of(pid)} for pid, p in self.postings.items()]
        return sorted(rows, key=lambda x: (x["status"] != "진행 중", x["company"], x["title"]))

    @staticmethod
    def make_scope(job=None, newcomer=False, posting_id=None):
        """화면의 선택을 평가셋과 같은 검색 범위 문자열로 바꾼다."""
        if posting_id:
            return f"공고 선택: {posting_id}"
        conds = [c for c in (job, "신입" if newcomer else None) if c]
        return "필터: " + " + ".join(conds) if conds else "전체"

    # ---- 한 질문 처리 ----------------------------------------------------
    def answer(self, question, scope="전체", structured_on_all=True):
        """질문 1개를 처리해 답변과 출처, 처리 방식, 토큰·비용을 돌려준다.
        structured_on_all=False면 조건 해석을 필터 범위에서만 쓴다(평가셋 재현용)."""
        t0 = time.time()
        opt = self.opt
        item = {"질문": question, "검색 범위": scope}
        ids = r.scope_ids(scope)  # None이면 전체
        tokens, mode, st, qvec = [], [], None, None
        embed_tokens = 0

        def embed():
            nonlocal qvec, embed_tokens
            if qvec is None:
                q = r.expand_query(question, self.groups)[0]
                res = self.client.embeddings.create(model=EMBED_MODEL, input=[q])
                qvec, embed_tokens = res.data[0].embedding, res.usage.total_tokens
            return qvec

        if scope.startswith("공고 선택:"):
            docs, metas = r.fetch_full_posting(self.col, ids[0])
            mode.append("공고 전체를 근거로 사용")
        else:
            if scope.startswith("필터:") or (structured_on_all and scope == "전체"):
                st, ptok = r.run_structured(item, ids if ids is not None else list(self.postings), self.col,
                                            self.client, opt["model"], opt["reasoning"])
                tokens.append((opt["model"], *ptok))
            if st:
                docs, metas = st["docs"], st["metas"]
                mode.append(f"조건으로 거름 → 공고 {len(st['matched'])}건")
            else:
                pp = r.per_posting_targets(item, ids, True, st)
                if pp:
                    docs, metas = r.query_per_posting(self.col, embed(), pp, opt["per_posting"], question)
                    mode.append(f"공고 {len(pp)}건에서 항목별로 고르게 검색")
                else:
                    where = {"posting_id": {"$in": ids}} if ids else None
                    res = self.col.query(query_embeddings=[embed()], n_results=opt["top_k"], where=where)
                    docs, metas = res["documents"][0], res["metadatas"][0]
                    mode.append(f"질문과 가까운 청크 {opt['top_k']}개 검색")
        retrieved = [(m["posting_id"], m["section_key"]) for m in metas]

        # 근거 만들기 (분류-본문 차이는 질문이 경력·신입을 물을 때만 붙인다)
        context = r.build_context(list(zip(docs, metas)))
        pids = list(dict.fromkeys(m["posting_id"] for m in metas))
        asked = bool(r.CAREER_ASK_RE.search(question))
        notes = r.build_diff_notes(pids, r.load_fields(), st["conds"] if st else None, st["refs"] if st else (), career=asked)
        base, structured_rules, diff_rules = self.rules
        blocks, system = [], base
        if st:
            blocks.append(f"[코드 필터 결과]\n{st['text']}")
            system += structured_rules
        if notes:
            blocks.append(notes)
            system += diff_rules
        user = "\n\n".join(blocks + [f"[근거]\n{context}", f"[질문]\n{question}"])
        ans, a_in, a_out = r.chat_json(self.client, opt["model"], system, user, r.ANSWER_SCHEMA, "answer", opt["reasoning"])
        tokens.append((opt["model"], a_in, a_out))

        # 출처: 답변이 표시한 공고 중 실제로 있는 공고만, 상태는 기준일로 계산
        sources, seen = [], set()
        for s in ans["sources"]:
            pid = s["posting_id"]
            if pid not in self.postings or (pid, s["section"]) in seen:
                continue
            seen.add((pid, s["section"]))
            p = self.postings[pid]
            sources.append({"posting_id": pid, "company": p["company"], "title": p["title"], "section": s["section"],
                            "closed_at": p["closed_at"], "status": self.status_of(pid), "url": p.get("url", "")})
        verify_text = r.normalize("\n\n".join(blocks + [context]))
        quotes = [q for q in ans["quotes"] if q.strip()]
        cost = embed_tokens / 1e6 * EMBED_PRICE_PER_1M
        for m, i, o in tokens:
            cost += i / 1e6 * r.PRICES[m][0] + o / 1e6 * r.PRICES[m][1]
        return {
            "question": question, "scope": scope, "answer": ans["answer"], "sources": sources,
            "llm_sources": ans["sources"], "quotes": quotes,
            "quotes_verified": sum(r.normalize(q) in verify_text for q in quotes),
            "mode": " · ".join(mode), "diff_notes": notes, "retrieved": retrieved,
            "tokens_in": sum(i for _, i, _ in tokens) + embed_tokens, "tokens_out": sum(o for _, _, o in tokens),
            "cost_usd": cost, "latency_ms": int((time.time() - t0) * 1000), "ref_date": self.ref_date,
            "closed_in_sources": [s["posting_id"] for s in sources if s["status"] != "진행 중"],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="eval")
    ap.add_argument("--scope", default="전체")
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    args = ap.parse_args()
    eng = Engine(args.dataset, top_k=args.top_k)
    out = eng.answer(args.question, args.scope)
    print(f"[{out['mode']}] {out['latency_ms']}ms, 약 ${out['cost_usd']:.5f}\n")
    print(out["answer"])
    print("\n출처:")
    for s in out["sources"]:
        print(f"- {s['company']} · {s['title']} · {s['section']} · 마감 {s['closed_at']} ({s['status']})")


if __name__ == "__main__":
    main()
