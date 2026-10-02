"""점핏 채용공고 청킹 스크립트 (3단계: 파싱·정제·청킹).

사용 문서(included=Y)를 읽어 docs/step3_cleaning_rules.md의 규칙대로 정제하고
두 가지 전략으로 청킹해서 저장한다.
- 전략 A: 항목(6개) 단위 그대로, 항목당 청크 1개
- 전략 B: A와 동일하되 800자 이상인 항목만 번호 소그룹/문단 경계로 하위 분할

데이터셋 (data/README.md > 평가용·서비스용 데이터 분리)
- eval (기본값): data/eval_snapshot_2026-09-30/postings.csv -> data/chunks/eval/
- service: data/postings.csv (최신 재수집본) -> data/chunks/service/

결과
- data/chunks/{데이터셋}/strategy_a.jsonl
- data/chunks/{데이터셋}/strategy_b.jsonl

사용법
    .venv\\Scripts\\python.exe src\\chunk_postings.py [eval|service]
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 평가는 9/30 스냅샷으로 고정, 서비스는 최신 postings.csv (data/README.md > 평가용·서비스용 데이터 분리)
DATASETS = {
    "eval": (ROOT / "data" / "eval_snapshot_2026-09-30" / "postings.csv", ROOT / "data" / "chunks" / "eval"),
    "service": (ROOT / "data" / "postings.csv", ROOT / "data" / "chunks" / "service"),
}

FIELDS = ["serviceInfo", "responsibility", "qualifications",
          "preferredRequirements", "welfares", "recruitProcess"]
SECTION_NAMES = {
    "serviceInfo": "회사소개", "responsibility": "주요업무", "qualifications": "자격요건",
    "preferredRequirements": "우대사항", "welfares": "복지", "recruitProcess": "채용절차",
}
# docs/step3_cleaning_rules.md > 빈 항목 판정 기준
PLACEHOLDERS = {"-", "없음", "해당없음", "해당사항 없음", "."}
LONG_THRESHOLD = 800  # docs/step3_cleaning_rules.md 청킹 전략 B 기준

# docs/step3_cleaning_rules.md > 예외 처리 방침: 대소문자 표기만 다듬는다 (동의어 병합 아님)
TECH_STACK_FIXES = {
    "etl": "ETL", "mfc": "MFC", "labview": "LabVIEW", "nosql": "NoSQL",
    "utm": "UTM", "vmware": "VMware", "vuex": "Vuex", "yolo": "YOLO",
}


def clean(text):
    """docs/step3_cleaning_rules.md 공통 규칙 1~6번 적용."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​‌‍﻿]", "", text)
    text = text.replace("\xa0", " ").replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    out, blank_run = [], 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip()


def is_empty(text):
    return not text or text in PLACEHOLDERS


def normalize_tech_stacks(raw_value):
    stacks = [s.strip() for s in (raw_value or "").split(",") if s.strip()]
    return [TECH_STACK_FIXES.get(s.lower(), s) for s in stacks]


def split_by_numbered_groups(text):
    """'1. 근무 환경' 같은 줄 앞에서 나눈다. 소그룹이 2개 미만이면 빈 리스트를 반환한다."""
    parts = re.split(r"\n(?=\d+\.\s)", text)
    return parts if len(parts) >= 2 else []


def split_by_paragraphs(text, limit):
    """빈 줄로 나뉜 문단을 limit자 넘지 않게 묶는다. 문단 자체가 limit보다 길면 그대로 한 조각이 된다."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return []
    parts, current = [], ""
    for para in paras:
        candidate = f"{current}\n\n{para}" if current else para
        if current and len(candidate) > limit:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts if len(parts) >= 2 else []


def split_by_lines(text, limit):
    """마지막 수단: 줄바꿈 기준으로 limit자 안 넘게 강제로 묶는다."""
    lines = text.split("\n")
    parts, current = [], ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if current and len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts if len(parts) >= 2 else [text]


def split_long_section(text):
    """docs/step3_cleaning_rules.md 청킹 전략 B: 번호 소그룹 → 문단 → 줄바꿈 순으로 시도.

    1차로 나눈 조각이 그래도 LONG_THRESHOLD 이상이면(예: 번호 소그룹 하나가 그 자체로 긴 경우)
    그 조각만 문단/줄바꿈 기준으로 한 번 더 나눈다.
    """
    parts = (split_by_numbered_groups(text)
             or split_by_paragraphs(text, LONG_THRESHOLD)
             or split_by_lines(text, LONG_THRESHOLD))

    result = []
    for part in parts:
        if len(part) >= LONG_THRESHOLD:
            result.extend(split_by_paragraphs(part, LONG_THRESHOLD)
                          or split_by_lines(part, LONG_THRESHOLD)
                          or [part])
        else:
            result.append(part)
    return result


def load_used_postings(postings_csv):
    with postings_csv.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["included"] == "Y"]


def build_chunks_for_posting(row, strategy):
    raw_path = ROOT / row["raw_path"]
    detail = json.loads(raw_path.read_text(encoding="utf-8"))
    collected_at = Path(row["raw_path"]).parts[-2]  # data/raw/{수집일}/{공고ID}.json
    job_categories = [c.strip() for c in row["job_categories"].split(",") if c.strip()]
    tech_stacks = normalize_tech_stacks(row["tech_stacks"])

    base_meta = {
        "posting_id": row["posting_id"],
        "company": row["company"],
        "title": row["title"],
        "job_categories": job_categories,
        "career_min": row["career_min"],
        "career_max": row["career_max"],
        "newcomer": row["newcomer"],
        "closed_at": row["closed_at"],
        "status": row["status"],
        "collected_at": collected_at,
        "tech_stacks": tech_stacks,
        "source_url": row["url"],
        "strategy": strategy,
    }

    chunks = []
    for field in FIELDS:
        text = clean(detail.get(field))
        if is_empty(text):
            continue
        section_kor = SECTION_NAMES[field]
        header_base = f"[{row['company']}/{row['title']}/{section_kor}]"

        if strategy == "B" and len(text) >= LONG_THRESHOLD:
            parts = split_long_section(text)
        else:
            parts = [text]

        total = len(parts)
        for i, part in enumerate(parts, start=1):
            header = header_base if total == 1 else f"[{row['company']}/{row['title']}/{section_kor} {i}/{total}]"
            body = part.strip()
            chunk = dict(base_meta)
            chunk.update({
                "chunk_id": f"{row['posting_id']}_{field}_{i}_{strategy}",
                "section": section_kor,
                "section_key": field,
                "chunk_index_in_section": i,
                "chunk_total_in_section": total,
                "char_count": len(body),
                "text": f"{header}\n{body}",
            })
            chunks.append(chunk)
    return chunks


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if dataset not in DATASETS:
        sys.exit(f"데이터셋은 {'/'.join(DATASETS)} 중 하나여야 합니다: {dataset}")
    postings_csv, chunks_dir = DATASETS[dataset]
    postings = load_used_postings(postings_csv)
    print(f"[{dataset}] {postings_csv.relative_to(ROOT).as_posix()} 사용 문서 {len(postings)}건 청킹 시작")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for strategy, filename in (("A", "strategy_a.jsonl"), ("B", "strategy_b.jsonl")):
        out_path = chunks_dir / filename
        total_chunks = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in postings:
                for chunk in build_chunks_for_posting(row, strategy):
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1
        print(f"  전략 {strategy}: 청크 {total_chunks}개 -> {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
