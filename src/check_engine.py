"""엔진 회귀 점검 (7단계): 평가셋 30문항을 rag_engine으로 돌려 run_eval의 6단계 결과와 맞는지 본다.

- 엔진은 평가셋과 같은 검색 범위로 부르되, 조건 해석은 필터 범위에서만 쓴다(structured_on_all=False) = run_eval과 같은 동작
- 채점은 run_eval과 같은 채점(채점 기준 v2)을 쓴다
- 비교 대상: 6단계 파이프라인 + top_k=10, 같은 설정 2회 (eval/results/2026-10-06_B_k10_..._pp2_p2_{a,b}.csv)

결과: eval/results/engine_check.csv (인용 열 없음)

사용법
    .venv\\Scripts\\python.exe src\\check_engine.py [--all-scope]
    --all-scope: 검색 범위가 전체인 문항에도 조건 해석을 켜서(앱의 동작) 어떻게 달라지는지 본다
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from build_index import log_usage
from rag_engine import Engine
import run_eval as r

COMPARE = ["2026-10-06_B_k10_gpt-6-luna_full_syn_struct_diffrel_pp2_p2_a.csv",
           "2026-10-06_B_k10_gpt-6-luna_full_syn_struct_diffrel_pp2_p2_b.csv"]
SCORE = {"정답": 1.0, "부분": 0.5, "오답": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-scope", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    eng = Engine("eval")
    items = r.load_csv(r.EVAL_SET)

    def one(item):
        out = eng.answer(item["질문"], item["검색 범위"], structured_on_all=args.all_scope)
        graded, (j_in, j_out) = r.grade(item, out["answer"], out["llm_sources"], eng.client, "gpt-6-luna")
        gold = r.gold_units(item["근거"])
        searched = "검색" in out["mode"] or "고르게" in out["mode"]
        hit = ""
        if gold and searched:
            hit = "O" if any(g in out["retrieved"] for g in gold) else "X"
        return {"id": item["id"], "분류": item["분류"], "질문": item["질문"], "검색 범위": item["검색 범위"],
                "처리 방식": out["mode"], "검색 적중": hit, **graded,
                "인용 검증": f"{out['quotes_verified']}/{len(out['quotes'])}", "답변 길이": len(out["answer"]),
                "응답 시간(ms)": out["latency_ms"], "비용($)": round(out["cost_usd"], 5),
                "_judge": (j_in, j_out)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, items))

    answerable = [x for x in rows if x["답변 판정"]]
    acc = sum(SCORE[x["답변 판정"]] for x in answerable) / len(answerable) * 100
    fresh = sum(SCORE[x["답변 판정"]] for x in rows if x["id"] in r.FRESHNESS_IDS) / len(r.FRESHNESS_IDS) * 100
    searched = [x for x in rows if x["검색 적중"]]
    hits = sum(x["검색 적중"] == "O" for x in searched)
    q_ok = sum(int(x["인용 검증"].split("/")[0]) for x in rows)
    q_all = sum(int(x["인용 검증"].split("/")[1]) for x in rows)
    none_rows = [x for x in rows if x["문서에 없음 처리"]]
    judge_cost = sum(i / 1e6 * r.PRICES["gpt-6-luna"][0] + o / 1e6 * r.PRICES["gpt-6-luna"][1] for i, o in (x["_judge"] for x in rows))
    cost = sum(x["비용($)"] for x in rows)
    print(f"답변 정확도 {acc:.1f}%, 최신성 {fresh:.1f}%, 출처 정확도 {q_ok / q_all * 100:.1f}% ({q_ok}/{q_all}), "
          f"검색 적중 {hits}/{len(searched)}, 문서에 없음 {sum(x['문서에 없음 처리'] == '정답' for x in none_rows)}/{len(none_rows)}")
    print(f"질문당 평균 비용 ${cost / len(rows):.5f}, 평균 응답 {sum(x['응답 시간(ms)'] for x in rows) / len(rows) / 1000:.1f}초")

    ref = [{x["id"]: x for x in r.load_csv(r.RAW_RESULTS_DIR / f)} for f in COMPARE]
    print("\n6단계(k=10) 두 실행과 판정이 다른 문항 (엔진 / 실행 a / 실행 b):")
    for x in rows:
        mine = x["답변 판정"] or x["문서에 없음 처리"]
        theirs = [(d[x["id"]]["답변 판정"] or d[x["id"]]["문서에 없음 처리"]) for d in ref]
        if len({mine, *theirs}) > 1:
            print(f"  {x['id']}: {mine} / {theirs[0]} / {theirs[1]}  | {x['처리 방식']}")
    ref_acc = [sum(SCORE[d[i]["답변 판정"]] for i in d if d[i]["답변 판정"]) / 25 * 100 for d in ref]
    print(f"기준(같은 설정 2회) 답변 정확도 {ref_acc[0]:.1f}% / {ref_acc[1]:.1f}%")

    out = r.RESULTS_DIR / ("engine_check_all_scope.csv" if args.all_scope else "engine_check.csv")
    cols = [k for k in rows[0] if not k.startswith("_")]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: x[k] for k in cols} for x in rows)
    total = cost + judge_cost
    log_usage({"시각": datetime.now().isoformat(timespec="seconds"), "작업": "check_engine", "모델": "gpt-6-luna",
               "입력 토큰": 0, "출력 토큰": 0, "비용(달러)": f"{total:.5f}",
               "메모": f"엔진 회귀 점검, 30문항{' (전체 범위에도 조건 해석)' if args.all_scope else ''}. 답변·조건 해석 ${cost:.4f} + 채점 ${judge_cost:.4f}"})
    print(f"\n→ {out.relative_to(r.ROOT).as_posix()}, 총 비용 약 ${total:.4f}")


if __name__ == "__main__":
    main()
