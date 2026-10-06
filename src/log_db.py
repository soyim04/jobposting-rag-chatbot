"""서비스 질문 로그 저장 (7단계, SQLite).

질문마다 한 줄: 질문·검색 범위·처리 방식·답변·출처·원문 인용·사용 옵션·토큰·비용·응답 시간·오류·사용자 피드백(👍/👎).
인용 문장에 공고 원문이 들어 있으므로 DB 파일(data/service_log.db)은 공개하지 않는다 (.gitignore).
공개용 요약은 summary()나 export_csv()(답변·인용 열 제외)로 만든다.

사용법
    .venv\\Scripts\\python.exe src\\log_db.py --summary
    .venv\\Scripts\\python.exe src\\log_db.py --export eval\\results\\service_log_public.csv
"""

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "service_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    dataset TEXT, ref_date TEXT,
    question TEXT NOT NULL,
    scope TEXT, mode TEXT,
    answer TEXT,
    sources TEXT,
    quotes TEXT,
    quotes_verified INTEGER, quotes_total INTEGER,
    closed_sources INTEGER,
    model TEXT, top_k INTEGER, per_posting INTEGER, prompt TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, latency_ms INTEGER,
    error TEXT,
    feedback INTEGER,
    feedback_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts);
"""
PUBLIC_COLUMNS = ["id", "ts", "dataset", "ref_date", "scope", "mode", "quotes_verified", "quotes_total", "closed_sources",
                  "model", "top_k", "per_posting", "prompt", "tokens_in", "tokens_out", "cost_usd", "latency_ms",
                  "error", "feedback"]


class LogDB:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, question, scope, out=None, options=None, dataset="", error=None):
        """엔진 결과(out) 또는 오류를 한 줄로 저장하고 로그 id를 돌려준다."""
        out = out or {}
        options = options or {}
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"), "dataset": dataset, "ref_date": out.get("ref_date"),
            "question": question, "scope": scope, "mode": out.get("mode"), "answer": out.get("answer"),
            "sources": json.dumps(out.get("sources", []), ensure_ascii=False),
            "quotes": json.dumps(out.get("quotes", []), ensure_ascii=False),
            "quotes_verified": out.get("quotes_verified"), "quotes_total": len(out.get("quotes", [])) if out else None,
            "closed_sources": len(out.get("closed_in_sources", [])) if out else None,
            "model": options.get("model"), "top_k": options.get("top_k"), "per_posting": options.get("per_posting"),
            "prompt": options.get("prompt"), "tokens_in": out.get("tokens_in"), "tokens_out": out.get("tokens_out"),
            "cost_usd": out.get("cost_usd"), "latency_ms": out.get("latency_ms"), "error": error,
        }
        cols = ", ".join(row)
        with self._conn() as conn:
            cur = conn.execute(f"INSERT INTO query_log ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
            return cur.lastrowid

    def set_feedback(self, log_id, value, note=None):
        """value: 1(도움이 됐다), -1(도움이 안 됐다), None(취소)"""
        with self._conn() as conn:
            conn.execute("UPDATE query_log SET feedback = ?, feedback_note = ? WHERE id = ?", (value, note, log_id))

    def recent(self, n=50):
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,))]

    def summary(self):
        with self._conn() as conn:
            q = lambda sql: conn.execute(sql).fetchone()[0]
            n = q("SELECT COUNT(*) FROM query_log")
            modes = {r["mode"] or "(오류)": r["c"] for r in conn.execute(
                "SELECT mode, COUNT(*) c FROM query_log GROUP BY mode ORDER BY c DESC")}
            return {
                "질문 수": n,
                "오류 수": q("SELECT COUNT(*) FROM query_log WHERE error IS NOT NULL"),
                "총 비용($)": round(q("SELECT COALESCE(SUM(cost_usd), 0) FROM query_log"), 5),
                "평균 비용($)": round(q("SELECT COALESCE(AVG(cost_usd), 0) FROM query_log WHERE error IS NULL"), 5),
                "평균 응답(초)": round(q("SELECT COALESCE(AVG(latency_ms), 0) FROM query_log WHERE error IS NULL") / 1000, 1),
                "도움이 됐다": q("SELECT COUNT(*) FROM query_log WHERE feedback = 1"),
                "도움이 안 됐다": q("SELECT COUNT(*) FROM query_log WHERE feedback = -1"),
                "마감 공고가 출처에 있던 질문": q("SELECT COUNT(*) FROM query_log WHERE closed_sources > 0"),
                "인용 확인 실패": q("SELECT COALESCE(SUM(quotes_total - quotes_verified), 0) FROM query_log"),
                "처리 방식별": modes,
            }

    def export_csv(self, path):
        """답변·인용·질문 원문을 뺀 공개용 CSV."""
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(f"SELECT {', '.join(PUBLIC_COLUMNS)} FROM query_log ORDER BY id")]
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PUBLIC_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--export", metavar="CSV")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    db = LogDB(args.db)
    if args.export:
        print(f"{db.export_csv(args.export)}건 → {args.export}")
    if args.summary or not args.export:
        for k, v in db.summary().items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
