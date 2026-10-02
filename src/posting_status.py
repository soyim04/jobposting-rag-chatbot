"""공고 마감 여부를 '오늘 날짜' 기준으로 판정한다 (서버 재수집 없이).

postings.csv와 청크 메타데이터의 status는 수집 시점의 상태라서, 수집 뒤 마감일이 지나도 '진행 중'으로 남는다.
검색·답변 단계에서는 저장된 status 대신 effective_status()로 판정한 값을 쓴다.

판정 규칙 (docs/step3_cleaning_rules.md > 마감 여부 판정)
1. 저장된 status가 '마감' 또는 '조기 마감'이면 그대로 쓴다 (재수집에서 이미 확인된 사실).
2. closed_at이 '상시'면 '진행 중'.
3. closed_at이 오늘보다 앞이면 '마감' (마감일 당일은 진행 중 — collect_jumpit.py와 같은 기준).
4. 그 외는 '진행 중'.
조기 마감(마감일 전에 내려간 공고)은 날짜만으로 알 수 없으므로 재수집해야 반영된다.

사용법 (오늘 기준 요약 출력)
    .venv\\Scripts\\python.exe src\\posting_status.py [YYYY-MM-DD]
"""

import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTINGS_CSV = ROOT / "data" / "postings.csv"

CLOSED_STATUSES = {"마감", "조기 마감"}


def effective_status(closed_at, stored_status, today=None):
    """오늘 기준 상태를 '진행 중' / '마감' / '조기 마감' 중 하나로 반환한다."""
    if stored_status in CLOSED_STATUSES:
        return stored_status
    if closed_at == "상시":
        return "진행 중"
    today = today or date.today().isoformat()
    return "마감" if closed_at < today else "진행 중"


def is_open(closed_at, stored_status, today=None):
    return effective_status(closed_at, stored_status, today) == "진행 중"


def main():
    today = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    with POSTINGS_CSV.open(encoding="utf-8-sig", newline="") as f:
        used = [r for r in csv.DictReader(f) if r["included"] == "Y"]

    counts = Counter(effective_status(r["closed_at"], r["status"], today) for r in used)
    print(f"기준일 {today} / 사용 문서 {len(used)}건: "
          + ", ".join(f"{k} {v}건" for k, v in sorted(counts.items())))

    changed = [r for r in used if effective_status(r["closed_at"], r["status"], today) != r["status"]]
    if changed:
        print(f"저장된 status와 다른 공고 {len(changed)}건 (마감일 지남):")
        for r in sorted(changed, key=lambda r: (r["closed_at"], r["posting_id"])):
            print(f"  {r['posting_id']}  {r['closed_at']}  {r['company']}")


if __name__ == "__main__":
    main()
