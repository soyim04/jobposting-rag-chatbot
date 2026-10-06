"""점핏 채용공고 Q&A 화면 (7단계, Streamlit).

사이드바에서 직무·경력 필터나 공고 하나를 고르고 질문하면, 6단계까지 실험으로 정한 파이프라인(src/rag_engine.py)이 공고를 근거로 답한다.
답변에는 출처 공고(회사·공고명·항목·마감일·상태·원문 링크)를 붙이고, 질문마다 src/log_db.py로 로그를 남긴다.
공고 원문은 재배포하지 않으므로 화면에는 인용 문장을 보여주지 않는다.

실행
    .venv\\Scripts\\python.exe -m streamlit run src\\app.py
환경 변수
    RAG_DATASET     eval 또는 service (기본: 서비스용 색인이 있으면 service, 없으면 eval)
    SERVICE_LOG_DB  로그 DB 경로 (기본 data/service_log.db)
"""

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from log_db import DB_PATH, LogDB
from rag_engine import DATASETS, Engine

st.set_page_config(page_title="점핏 채용공고 Q&A", page_icon="🔎", layout="wide")

JOBS = {"전체": None, "백엔드": "백엔드", "프론트엔드": "프론트엔드"}


def pick_dataset():
    forced = os.environ.get("RAG_DATASET")
    if forced in DATASETS:
        return forced
    cfg = DATASETS["service"]
    if cfg["fields"].exists():
        try:
            import chromadb
            import run_eval as r
            chromadb.PersistentClient(path=str(r.CHROMA_DIR)).get_collection(cfg["collection"])
            return "service"
        except Exception:
            pass
    return "eval"


def code_version():
    """엔진이 쓰는 코드·사전 파일의 수정 시각. 바뀌면 캐시된 엔진을 새로 만든다(코드를 고친 뒤 서버를 다시 켤 필요 없게)."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    files = list((root / "src").glob("*.py")) + [root / "data" / "tech_synonyms.json"]
    return max(f.stat().st_mtime for f in files)


@st.cache_resource
def _build_engine(dataset, version):
    return Engine(dataset)


def get_engine(dataset):
    return _build_engine(dataset, code_version())


@st.cache_resource
def get_db():
    return LogDB(os.environ.get("SERVICE_LOG_DB") or DB_PATH)


def save_feedback(log_id, key):
    """st.feedback('thumbs') 값: 1이 좋아요, 0이 별로예요, None은 취소"""
    v = st.session_state.get(key)
    get_db().set_feedback(log_id, None if v is None else (1 if v == 1 else -1))


def render_answer(entry):
    out = entry["out"]
    st.markdown(out["answer"])
    if out["closed_in_sources"]:
        st.warning("마감된 공고가 근거에 포함되어 있습니다. 마감된 공고는 지원할 수 없어요.")
    grouped = {}
    for s in out["sources"]:  # 같은 공고의 여러 항목은 한 줄로 묶는다
        grouped.setdefault(s["posting_id"], {**s, "sections": []})["sections"].append(s["section"])
    with st.expander(f"근거 공고 {len(grouped)}건"):
        if not grouped:
            st.write("답변이 표시한 근거 공고가 없습니다.")
        for s in grouped.values():
            status = s["status"] if s["status"] == "진행 중" else f":red[{s['status']}]"
            link = f" · [점핏에서 보기]({s['url']})" if s["url"] else ""
            st.markdown(f"- **{s['company']}** · {s['title']} · {', '.join(s['sections'])} · 마감 {s['closed_at']} ({status}){link}")
    st.caption(f"처리 방식: {out['mode']} · {out['latency_ms'] / 1000:.1f}초 · 약 ${out['cost_usd']:.4f}"
               f" · 기준일 {out['ref_date']}")
    key = f"fb_{entry['log_id']}"
    st.feedback("thumbs", key=key, on_change=save_feedback, args=(entry["log_id"], key))


def run_question(engine, db, question, scope, scope_label):
    entry = {"q": question, "scope_label": scope_label}
    try:
        out = engine.answer(question, scope)
        entry["out"] = out
        entry["log_id"] = db.log(question, scope, out, engine.opt, engine.dataset)
    except Exception as e:  # 화면은 계속 동작하게 두고 오류도 로그에 남긴다
        entry["error"] = "답변을 만들지 못했습니다. 잠시 뒤 다시 시도해 주세요."
        db.log(question, scope, None, engine.opt, engine.dataset, error=f"{type(e).__name__}: {e}")
    st.session_state.history.append(entry)


def sidebar(engine):
    st.sidebar.header("검색 범위")
    job_label = st.sidebar.selectbox("직무", list(JOBS), help="점핏 직무 분류 기준입니다.")
    newcomer = st.sidebar.checkbox("신입 공고만", help="점핏 경력 분류가 신입인 공고만 봅니다.")
    filter_scope = Engine.make_scope(JOBS[job_label], newcomer)
    import run_eval as r
    ids = r.scope_ids(filter_scope)
    show_closed = st.sidebar.checkbox("마감된 공고도 목록에 포함", help="끄면 진행 중인 공고만 고를 수 있습니다.")
    options = [o for o in engine.posting_options()
               if (ids is None or o["posting_id"] in ids) and (show_closed or o["status"] == "진행 중")]
    labels = {o["posting_id"]: f"{o['company']} · {o['title']} · 마감 {o['closed_at']} ({o['status']})" for o in options}
    pid = st.sidebar.selectbox("공고 하나만 골라 묻기", [None] + list(labels), format_func=lambda x: "선택 안 함" if x is None else labels[x],
                               help="공고를 고르면 그 공고 전체를 근거로 답합니다.")
    scope = Engine.make_scope(JOBS[job_label], newcomer, pid)
    label = labels[pid] if pid else (f"{job_label}" + (" · 신입" if newcomer else "") if scope != "전체" else "전체 공고")
    st.sidebar.caption(f"선택한 범위: {label} ({len(options)}건)" if not pid else f"선택한 공고: {label}")
    st.sidebar.divider()
    total = sum(e["out"]["cost_usd"] for e in st.session_state.history if "out" in e)
    st.sidebar.caption(f"이번 접속의 질문 {len(st.session_state.history)}개, 비용 약 ${total:.4f}")
    return scope, label


def examples(engine):
    ref = date.fromisoformat(engine.ref_date) + timedelta(days=10)
    return ["재택이나 유연근무 되는 곳 있어?", "신입 백엔드 공고에서 공통으로 요구하는 게 뭐야?",
            "Node.js로 지원할 수 있는 곳 있어?", "4년제 졸업 안 해도 되는 곳 있어?",
            f"{ref.month}월 {ref.day}일 전에 마감되는 공고 있어?"]


def log_tab(db):
    s = db.summary()
    c = st.columns(4)
    c[0].metric("질문 수", s["질문 수"])
    c[1].metric("총 비용($)", s["총 비용($)"])
    c[2].metric("평균 응답(초)", s["평균 응답(초)"])
    c[3].metric("👍 / 👎", f"{s['도움이 됐다']} / {s['도움이 안 됐다']}")
    st.caption(f"오류 {s['오류 수']}건 · 마감 공고가 출처에 있던 질문 {s['마감 공고가 출처에 있던 질문']}건 · 인용 확인 실패 {s['인용 확인 실패']}건")
    st.write("처리 방식별", s["처리 방식별"])
    rows = db.recent(100)
    if rows:
        df = pd.DataFrame(rows)[["id", "ts", "question", "scope", "mode", "cost_usd", "latency_ms", "feedback", "error"]]
        st.dataframe(df, hide_index=True, width="stretch")


def main():
    dataset = pick_dataset()
    engine, db = get_engine(dataset), get_db()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)

    st.title("점핏 채용공고 Q&A")
    n_open = sum(1 for o in engine.posting_options() if o["status"] == "진행 중")
    st.caption(f"점핏 개발자 채용공고 {len(engine.postings)}건(진행 중 {n_open}건), 기준일 {engine.ref_date}"
               + ("" if dataset == "service" else " · 평가용 데이터(9/30 수집)로 동작 중입니다"))
    scope, scope_label = sidebar(engine)

    tab_chat, tab_log = st.tabs(["질문하기", "로그"])
    with tab_chat:
        st.caption("점핏 공고 내용만 근거로 답하고, 근거 공고와 마감 여부를 함께 보여줍니다. 공고 원문은 점핏에서 확인하세요.")
        if not st.session_state.history:
            cols = st.columns(len(examples(engine)))
            for col, q in zip(cols, examples(engine)):
                if col.button(q, width="stretch"):
                    st.session_state.pending = q
        for e in st.session_state.history:
            with st.chat_message("user"):
                st.write(e["q"])
                st.caption(f"범위: {e['scope_label']}")
            with st.chat_message("assistant"):
                if "error" in e:
                    st.error(e["error"])
                else:
                    render_answer(e)
    with tab_log:
        log_tab(db)

    question = st.chat_input("공고에 대해 물어보세요") or st.session_state.pending
    if question:
        st.session_state.pending = None
        with tab_chat:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("공고를 찾아 읽는 중..."):
                    run_question(engine, db, question, scope, scope_label)
        st.rerun()


main()
