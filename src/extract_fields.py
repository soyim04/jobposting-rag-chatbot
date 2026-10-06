"""공고별 구조화 칸 추출 (6단계 3단계: 조건 찾기를 코드로).

평가용 청크(`data/chunks/eval/strategy_b.jsonl`, 9/30 스냅샷)에서 공고마다 다음 칸을 뽑는다.
- tech_required / tech_preferred : 자격요건·우대사항 본문에 나온 기술 (규칙: data/tech_synonyms.json 사전, 대표어로 저장)
- test_code                      : 테스트 코드 작성 경험 언급 위치 (required=자격요건, preferred=우대사항) (규칙)
- coding_test                    : 채용절차에 코딩테스트·과제 전형 언급 (규칙)
- remote_work / flexible_hours   : 재택·원격·하이브리드 / 유연·자율·시차 출퇴근 (LLM 1회 추출, 고정 스키마)
점핏이 붙인 기술스택 태그(overview의 tech_stacks)와 별개로 본문 기준 값을 뽑는다. 둘이 다른 공고는 4단계 재료다.

결과
- data/posting_fields_eval.csv           : 공고별 칸 (공개, 불리언·분류값만)
- data/raw/posting_fields_evidence.csv   : LLM이 근거로 복사한 원문 문장 (비공개, 커밋 금지)
- eval/usage_log.csv                     : 사용 토큰과 비용

사용법
    .venv\\Scripts\\python.exe src\\extract_fields.py
"""

import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from build_index import log_usage
from run_eval import (PRICES, ROOT, SECTION_KOR, chat_json, load_csv, load_synonym_groups,
                      normalize, term_in_text)

CHUNKS = ROOT / "data" / "chunks" / "eval" / "strategy_b.jsonl"
SNAPSHOT = ROOT / "data" / "eval_snapshot_2026-09-30" / "postings.csv"
OUT = ROOT / "data" / "posting_fields_eval.csv"
EVIDENCE = ROOT / "data" / "raw" / "posting_fields_evidence.csv"
MODEL = "gpt-6-luna"
NOT_TECH = {"테스트 코드", "CI/CD"}  # 기술 목록에서는 뺀다 (개념 그룹)
CODING_TEST_RE = re.compile(r"코딩\s*테스트|라이브\s*코딩|알고리즘\s*테스트|과제")
TEST_CODE_RE = re.compile(r"테스트\s*(코드|작성)")

REMOTE_SYSTEM = """너는 채용공고 본문에서 근무 형태를 확인하는 추출기다. 본문에 명시된 것만 true로 한다.
- remote_work: 재택근무·원격근무·리모트·하이브리드 근무가 가능하다고 적힌 경우
- flexible_hours: 유연근무제·자율 출퇴근·시차출퇴근·선택적 근로시간제·출근 시간 선택처럼 출퇴근 시간을 고를 수 있다고 적힌 경우
- 근무 장소 안내, 일반적인 문화 소개(예: 자율적인 분위기), 불가·없음이라는 문장은 해당하지 않는다. 추측하지 않는다.
- evidence: 판단 근거가 된 문장을 본문에서 글자 그대로 복사한 목록 (둘 다 false면 빈 목록)"""

REMOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "remote_work": {"type": "boolean"},
        "flexible_hours": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["remote_work", "flexible_hours", "evidence"],
    "additionalProperties": False,
}


def load_sections():
    """공고ID → {항목키: 본문 텍스트} (청크 순서대로 이어 붙임, 청크 첫 줄의 [회사/공고명/항목] 머리말은 뺀다)."""
    chunks = defaultdict(list)
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        chunks[c["posting_id"]].append(c)
    sections = {}
    for pid, cs in chunks.items():
        by_key = defaultdict(list)
        for c in sorted(cs, key=lambda c: c["chunk_index_in_section"]):
            by_key[c["section_key"]].append(c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"])
        sections[pid] = {k: "\n".join(v) for k, v in by_key.items()}
    return sections


def find_techs(text, groups):
    return [g[0] for g in groups if g[0] not in NOT_TECH and any(term_in_text(t, text) for t in g)]


def rule_fields(sec, groups):
    qual, pref, proc = sec.get("qualifications", ""), sec.get("preferredRequirements", ""), sec.get("recruitProcess", "")
    test_groups = next(g for g in groups if g[0] == "테스트 코드")

    def has_test(text):
        return bool(TEST_CODE_RE.search(text)) or any(term_in_text(t, text) for t in test_groups)

    return {
        "tech_required": ";".join(find_techs(qual, groups)),
        "tech_preferred": ";".join(find_techs(pref, groups)),
        "test_code": "required" if has_test(qual) else ("preferred" if has_test(pref) else ""),
        "coding_test": "Y" if CODING_TEST_RE.search(proc) else "N",
    }


def llm_fields(pid, sec, client):
    body = "\n\n".join(f"[{SECTION_KOR[k]}]\n{sec[k]}" for k in SECTION_KOR if k in sec and k != "overview")
    out, t_in, t_out = chat_json(client, MODEL, REMOTE_SYSTEM, body, REMOTE_SCHEMA, "remote", None)
    norm_body = normalize(body)
    evidence = [e for e in out["evidence"] if e.strip()]
    return pid, out, evidence, sum(normalize(e) in norm_body for e in evidence), t_in, t_out


def main():
    load_dotenv(ROOT / ".env")
    client = OpenAI()
    groups = load_synonym_groups()
    sections = load_sections()
    postings = [r for r in load_csv(SNAPSHOT) if r["included"] == "Y"]
    ids = [r["posting_id"] for r in postings]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = {r[0]: r for r in pool.map(lambda pid: llm_fields(pid, sections[pid], client), ids)}

    rows, evid = [], []
    for p in postings:
        pid = p["posting_id"]
        _, out, evidence, matched, _, _ = results[pid]
        rows.append({"posting_id": pid, "company": p["company"], **rule_fields(sections[pid], groups),
                     "remote_work": "Y" if out["remote_work"] else "N",
                     "flexible_hours": "Y" if out["flexible_hours"] else "N"})
        evid.append({"posting_id": pid, "company": p["company"], "근거 문장": " | ".join(evidence),
                     "원문 일치": f"{matched}/{len(evidence)}"})
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT, rows), (EVIDENCE, evid)):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    t_in = sum(r[4] for r in results.values())
    t_out = sum(r[5] for r in results.values())
    price_in, price_out = PRICES[MODEL]
    cost = t_in / 1e6 * price_in + t_out / 1e6 * price_out
    log_usage({"시각": datetime.now().isoformat(timespec="seconds"), "작업": "extract_fields", "모델": MODEL,
               "입력 토큰": t_in, "출력 토큰": t_out, "비용(달러)": f"{cost:.5f}", "메모": f"재택·유연근무 추출, {len(ids)}건"})
    print(f"{len(rows)}건 → {OUT.relative_to(ROOT).as_posix()}, 비용 약 ${cost:.4f}")
    for k in ("remote_work", "flexible_hours", "coding_test"):
        print(f"  {k}=Y: {sum(r[k] == 'Y' for r in rows)}건")
    for k in ("tech_required", "tech_preferred"):
        print(f"  {k} 비어 있음: {sum(not r[k] for r in rows)}건")
    print(f"  test_code: required {sum(r['test_code'] == 'required' for r in rows)}, "
          f"preferred {sum(r['test_code'] == 'preferred' for r in rows)}")
    bad = [e["posting_id"] for e in evid if e["원문 일치"].split("/")[0] != e["원문 일치"].split("/")[1]]
    print(f"  근거 문장이 원문과 일치하지 않는 공고: {bad or '없음'}")


if __name__ == "__main__":
    main()
