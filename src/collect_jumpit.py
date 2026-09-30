"""점핏(Jumpit) 개발 직군 채용공고 수집 스크립트.

실행할 때마다 현재 진행 중인 공고를 다시 조회해서
- 새 공고는 추가하고, 내용이 바뀐 공고는 새 스냅샷으로 저장하며
- 목록에서 사라진 공고는 삭제하지 않고 상태만 "마감"으로 바꾼다.

결과
- data/raw/{수집일}/{공고ID}.json : 원문 스냅샷 (저장소에 올리지 않음)
- data/postings.csv               : 공고별 최신 상태와 사용/제외 여부
- data/collection_summary.csv     : 실행할 때마다 한 줄씩 쌓이는 수집 요약

사용법
    python src/collect_jumpit.py
"""

import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
POSTINGS_CSV = ROOT / "data" / "postings.csv"
SUMMARY_CSV = ROOT / "data" / "collection_summary.csv"

LIST_URL = "https://jumpit-api.saramin.co.kr/api/positions"
DETAIL_URL = "https://jumpit-api.saramin.co.kr/api/position/{id}"
POSTING_URL = "https://jumpit.saramin.co.kr/position/{id}"

# 수집 범위: 백엔드·프론트엔드·풀스택 중 최소 경력 3년 이하 (신입~3년차 타겟)
JOB_CATEGORIES = {1: "서버/백엔드", 2: "프론트엔드", 3: "웹 풀스택"}
MAX_CAREER_START = 3

# 주요업무·자격요건이 비어 있거나 본문이 이보다 짧으면 답변 근거가 없어 제외
MIN_BODY_CHARS = 100
REQUIRED_FIELDS = {"responsibility": "주요업무", "qualifications": "자격요건"}
TEXT_FIELDS = ["serviceInfo", "responsibility", "qualifications",
               "preferredRequirements", "welfares", "recruitProcess"]

REQUEST_INTERVAL = 0.7  # 초. 서버 부하를 줄이기 위해 요청마다 쉰다
MAX_PAGES = 50
KST = timezone(timedelta(hours=9))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational project: jobposting-rag-chatbot)",
    "Referer": "https://jumpit.saramin.co.kr/",
}

COLUMNS = [
    "posting_id", "source", "company", "title", "job_categories",
    "career_min", "career_max", "newcomer", "education", "location",
    "tech_stacks", "published_at", "closed_at", "status",
    "first_collected", "last_seen", "content_hash", "char_count",
    "included", "exclude_reason", "raw_path", "url",
]

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url, params=None):
    for attempt in range(3):
        try:
            res = session.get(url, params=params, timeout=15)
            res.raise_for_status()
            return res.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
        finally:
            time.sleep(REQUEST_INTERVAL)


def clean(text):
    return (text or "").replace("​", "").replace("\r", "").strip()


def fetch_listing():
    """직무별 목록을 끝까지 넘기며 진행 중인 공고를 모은다."""
    found = {}
    for cat_id, cat_name in JOB_CATEGORIES.items():
        for page in range(1, MAX_PAGES + 1):
            data = get_json(LIST_URL, {"sort": "rsp_rate", "highlight": "false",
                                       "jobCategory": cat_id, "page": page})
            positions = data["result"]["positions"]
            if not positions:
                break
            for p in positions:
                found[p["id"]] = p
        print(f"  목록 조회: {cat_name} 누적 {len(found)}건")
    return found


def content_hash(detail):
    parts = {k: clean(detail.get(k)) for k in TEXT_FIELDS}
    parts["title"] = detail.get("title")
    parts["closedAt"] = detail.get("closedAt")
    parts["techStacks"] = [t["stack"] for t in detail.get("techStacks") or []]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def load_postings():
    if not POSTINGS_CSV.exists():
        return {}
    with POSTINGS_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {int(r["posting_id"]): r for r in csv.DictReader(f)}


def save_postings(rows):
    order = {"진행 중": 0, "조기 마감": 1, "마감": 2}
    sorted_rows = sorted(rows.values(), key=lambda r: (order.get(r["status"], 9), int(r["posting_id"])))
    with POSTINGS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_rows)


def append_summary(summary):
    is_new = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if is_new:
            writer.writeheader()
        writer.writerow(summary)


def main():
    today = datetime.now(KST).date().isoformat()
    print(f"[{today}] 점핏 공고 수집 시작")

    listing = fetch_listing()
    rows = load_postings()
    stats = {"신규": 0, "변경": 0, "변경 없음": 0, "마감 처리": 0}

    in_scope = {pid: p for pid, p in listing.items() if p["minCareer"] <= MAX_CAREER_START}
    print(f"  수집 대상(최소 경력 {MAX_CAREER_START}년 이하): {len(in_scope)}건 / 전체 {len(listing)}건")

    for i, (pid, p) in enumerate(listing.items(), 1):
        row = rows.get(pid) or {"posting_id": pid, "source": "jumpit", "first_collected": today}
        row.update({
            "company": p["companyName"], "title": p["title"],
            "job_categories": p.get("jobCategory", ""),
            "career_min": p["minCareer"], "career_max": p["maxCareer"],
            "newcomer": "Y" if p.get("newcomer") else "N",
            "closed_at": (p.get("closedAt") or "상시")[:10],
            "status": "진행 중", "last_seen": today,
            "url": POSTING_URL.format(id=pid),
        })

        if pid not in in_scope:
            row.update({"included": "N", "exclude_reason": f"경력 범위 밖 (최소 {p['minCareer']}년)"})
            rows[pid] = row
            continue

        detail = get_json(DETAIL_URL.format(id=pid))["result"]
        h = content_hash(detail)
        if h != row.get("content_hash"):
            stats["변경" if row.get("content_hash") else "신규"] += 1
            snap = RAW_DIR / today / f"{pid}.json"
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
            row["raw_path"] = snap.relative_to(ROOT).as_posix()
        else:
            stats["변경 없음"] += 1

        chars = sum(len(clean(detail.get(k))) for k in TEXT_FIELDS)
        places = detail.get("workingPlaces") or []
        row.update({
            "education": detail.get("educationName") or "",
            "location": places[0]["address"] if places else "",
            "tech_stacks": ", ".join(t["stack"] for t in detail.get("techStacks") or []),
            "published_at": (detail.get("publishedAt") or "")[:10],
            "closed_at": (detail.get("closedAt") or "상시")[:10],
            "content_hash": h, "char_count": chars,
        })
        missing = [name for key, name in REQUIRED_FIELDS.items() if not clean(detail.get(key))]
        if missing:
            row.update({"included": "N", "exclude_reason": f"필수 항목 미기재 ({'/'.join(missing)})"})
        elif chars < MIN_BODY_CHARS:
            row.update({"included": "N", "exclude_reason": f"본문 {MIN_BODY_CHARS}자 미만"})
        else:
            row.update({"included": "Y", "exclude_reason": ""})
        rows[pid] = row
        if i % 20 == 0:
            print(f"  상세 조회 진행: {i}/{len(listing)}")

    # 목록에서 사라진 공고는 지우지 않고 상태만 바꾼다
    for pid, row in rows.items():
        if pid not in listing and row["status"] == "진행 중":
            passed = row["closed_at"] != "상시" and row["closed_at"] < today
            row["status"] = "마감" if passed else "조기 마감"
            stats["마감 처리"] += 1

    save_postings(rows)

    used = [r for r in rows.values() if r["included"] == "Y"]
    excluded = [r for r in rows.values() if r["included"] == "N"]
    reasons = {}
    for r in excluded:
        key = r["exclude_reason"].split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    summary = {
        "수집일": today, "출처": "점핏",
        "목록 공고 수": len(listing), "수집 대상": len(in_scope),
        **stats,
        "누적 문서 수": len(rows),
        "사용 문서 수": len(used),
        "사용 문서 중 진행 중": sum(r["status"] == "진행 중" for r in used),
        "제외 문서 수": len(excluded),
        "제외 사유": "; ".join(f"{k} {v}건" for k, v in reasons.items()),
        "사용 문서 총 글자 수": sum(int(r["char_count"] or 0) for r in used),
    }
    append_summary(summary)

    print("\n수집 요약")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
