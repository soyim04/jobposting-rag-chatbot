"""평가셋 자동 실행 스크립트 (4단계).

eval/eval_set.csv의 30문항을 검색 → 답변 생성 → 채점하고 채점표를 저장한다.
채점 기준은 eval/README.md를 따른다.
- 코드로 채점: 검색 적중률·근거 재현율, 근거 표시 정확도, 출처 정확도(인용 문장 실재 여부)
- LLM으로 채점: 답변 정확도, 문서에 없는 질문 처리 (`사람 판정` 열은 사람이 채워 일치율을 계산)

결과
- eval/results/{실행일}_{전략}_k{top_k}_{모델}.csv : 문항별 채점표 (공개용, 공고 원문 인용 열 제외)
- eval/results/raw/ 같은 이름                    : 인용 열까지 포함한 전체 채점표 (비공개, .gitignore)
- eval/results/summary.csv                        : 실행마다 지표 한 줄 (실험 로그)
- eval/usage_log.csv                              : 사용 토큰과 비용

사용법
    .venv\\Scripts\\python.exe src\\run_eval.py                    # 전략 A·B, top-5
    .venv\\Scripts\\python.exe src\\run_eval.py --strategy B --ids A01 N04
    .venv\\Scripts\\python.exe src\\run_eval.py --regrade 2026-10-02_B_k5_gpt-6-luna.csv   # 답변 재사용, 다시 채점
"""

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from build_index import EMBED_MODEL, EMBED_PRICE_PER_1M, collection_name, log_usage
from posting_status import effective_status

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "eval" / "eval_set.csv"
SNAPSHOT = ROOT / "data" / "eval_snapshot_2026-09-30" / "postings.csv"
RESULTS_DIR = ROOT / "eval" / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"  # 공고 원문 인용이 들어 있어 커밋하지 않는다
PRIVATE_COLUMNS = ("인용",)
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
SYNONYMS_JSON = ROOT / "data" / "tech_synonyms.json"
CHROMA_DIR = ROOT / "chroma_db"

REF_DATE = "2026-10-02"  # eval/README.md > 평가 기준일
# 달러/100만 토큰 (입력, 출력). OpenAI 가격표 2026-10-02 확인
PRICES = {"gpt-6-luna": (0.10, 0.50), "gpt-5.6-luna": (0.20, 1.20), "gpt-4.1-mini": (0.40, 1.60),
          "gpt-6.1-sol": (2.00, 10.00)}
SECTION_KOR = {"overview": "공고 개요", "serviceInfo": "회사소개", "responsibility": "주요업무",
               "qualifications": "자격요건", "preferredRequirements": "우대사항",
               "welfares": "복지", "recruitProcess": "채용절차"}
VERDICT_SCORE = {"정답": 1.0, "부분": 0.5, "오답": 0.0}
FRESHNESS_IDS = {"A19", "A20", "T04"}
PROMPT_VERSION = "v1"
GRADING_VERSION = "v4"  # v1: 근거 표시를 공고ID만 비교 / v2: 공고ID+항목 비교 / v4: 정답지 수정(A03·A04·A09·A14, 질문만 보고) (eval/README.md > 채점 기준 변경 기록)
KOR_TO_SECTION = {v: k for k, v in SECTION_KOR.items()}

ANSWER_SYSTEM = f"""너는 점핏 개발자 채용공고를 근거로 답하는 상담 챗봇이다. 오늘은 {REF_DATE}이다.

규칙
1. 아래 [근거]에 있는 내용만으로 답한다. 근거에 없는 내용은 "확인되지 않습니다"라고 답하고, 추측하거나 외부 지식으로 채우지 않는다.
2. 질문을 잇기 위해 일반 지식(예: 프레임워크가 어떤 언어 기반인지)을 써야 하면, 그 부분이 공고 내용이 아니라 추론이라고 밝힌다.
3. 공고를 언급할 때마다 회사명과 공고명을 쓰고, 마감일과 상태는 [근거]에 적힌 값을 그대로 옮긴다.
4. 마감된 공고는 지원할 수 없다고 분명히 말한다.
5. 서로 다른 공고의 내용을 섞지 않는다. 같은 회사라도 공고ID가 다르면 다른 공고다.
6. 질문에 대한 답만 간결하게 한다.

출력 필드
- answer: 사용자에게 보여 줄 답변
- sources: 답변의 근거로 쓴 공고ID와 항목 (근거를 쓰지 않았으면 빈 목록)
- quotes: 답변의 근거가 된 원문 문장을 [근거]에서 글자 그대로 복사한 목록"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {
            "type": "object",
            "properties": {"posting_id": {"type": "string"}, "section": {"type": "string"}},
            "required": ["posting_id", "section"], "additionalProperties": False}},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "sources", "quotes"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """너는 채용공고 챗봇의 답변을 채점하는 채점자다. [문항]의 기준으로 [챗봇 답변]을 판정한다.

판정 규칙
- 정답: 필수 요소를 모두 말함. 부분: 일부만 말함. 오답: 필수 요소를 말하지 못함.
- "섞이면 안 되는 내용"이 답변에 하나라도 들어가면 필수 요소와 관계없이 오답.
- 질문 밖 정보: 공고와 맞으면 감점 없음, 틀리면 오답.
- 일반 지식 추론: 추론이라고 구분하지 않고 공고 내용처럼 말하면 오답.
- 채점 메모가 있으면 그 기준을 우선한다.
- 분류가 "문서에 없음"인 문항: "확인되지 않습니다"처럼 모른다고 답하고 다른 공고 내용으로 지어내지 않으면 정답, 지어내면 오답.
- 표현이 달라도 뜻이 같으면 맞은 것으로 본다."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["정답", "부분", "오답"]},
        "missing": {"type": "array", "items": {"type": "string"}},
        "contaminated": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "missing", "contaminated", "reason"],
    "additionalProperties": False,
}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


POSTINGS = {r["posting_id"]: r for r in load_csv(SNAPSHOT) if r["included"] == "Y"}


def scope_ids(scope):
    """eval/README.md > 필터 정의. None이면 전체."""
    if scope == "전체":
        return None
    if scope.startswith("공고 선택:"):
        return [scope.split(":", 1)[1].strip()]
    conds = [c.strip() for c in scope.split(":", 1)[1].split("+")]
    ids = []
    for pid, p in POSTINGS.items():
        cats = p["job_categories"]
        ok = True
        for c in conds:
            if c == "신입":
                ok &= p["newcomer"] == "Y"
            elif c == "백엔드":
                ok &= "서버/백엔드 개발자" in cats
            elif c == "프론트엔드":
                ok &= "프론트엔드 개발자" in cats
            else:
                raise ValueError(f"알 수 없는 필터 조건: {c}")
        if ok:
            ids.append(pid)
    return sorted(ids)


def gold_units(evidence):
    """'공고ID/항목키' 목록. '(없음)'과 '참고:' 항목은 정답 근거가 아니다."""
    units = []
    for ref in evidence.split(";"):
        ref = ref.strip()
        if not ref or ref.startswith("(없음)") or ref.startswith("참고:"):
            continue
        units.append(tuple(ref.split("/")))
    return units


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def build_context(hits):
    blocks = []
    for i, (doc, meta) in enumerate(hits, start=1):
        status = effective_status(meta["closed_at"], POSTINGS[meta["posting_id"]]["status"], REF_DATE)  # 저장 상태(조기 마감)도 반영
        body = doc.split("\n", 1)[1] if "\n" in doc else doc
        blocks.append(
            f"[근거 {i}] 공고ID {meta['posting_id']} | {meta['company']} · {meta['title']} | 항목: {meta['section']}\n"
            f"마감일 {meta['closed_at']} · 상태: {status} (기준일 {REF_DATE})\n{body}")
    return "\n\n".join(blocks)


def load_synonym_groups():
    """data/tech_synonyms.json → [[대표어, 표기...], ...]. '_'로 시작하는 키(설명·관련 기술)는 쓰지 않는다."""
    data = json.loads(SYNONYMS_JSON.read_text(encoding="utf-8"))
    return [[k, *v] for k, v in data.items() if not k.startswith("_")]


KOR_PARTICLES = ("을", "를", "이", "가", "은", "는", "의", "로", "에", "도", "만", "과", "와", "랑", "으로", "에서",
                 "경험", "관련", "개발", "써", "쓰")


def term_in_text(term, text):
    """영문 표기는 단어 경계로만 일치(Java가 JavaScript에 걸리지 않게), 3자 이하 영문은 대소문자도 구분.
    한글 표기는 2자 이상이고, 바로 뒤가 한글이 아니거나 조사·어미일 때만 일치로 본다
    ('뷰'가 '인터뷰'에, '인공지능'이 회사명 '인공지능팩토리'에 걸리지 않게)."""
    if re.search(r"[가-힣]", term):
        if len(term) < 2:
            return False
        for m in re.finditer(re.escape(term), text):
            rest = text[m.end():]
            if not rest or not re.match(r"[가-힣]", rest) or rest.startswith(KOR_PARTICLES):
                return True
        return False
    flags = 0 if len(term) <= 3 else re.IGNORECASE
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, flags) is not None


def expand_query(question, groups):
    """2단계: 질문에 기술 표기가 있으면 같은 기술의 다른 표기를 덧붙여 검색용 질문을 만든다. (확장 질문, 덧붙인 표기)"""
    extra = []
    for terms in groups:
        if any(term_in_text(t, question) for t in terms):
            extra += [t for t in terms if t not in extra and t.lower() not in question.lower()]
    return (f"{question} ({', '.join(extra)})" if extra else question), extra


def chat_json(client, model, system, user, schema, name, reasoning):
    kwargs = {"reasoning_effort": reasoning} if reasoning else {}
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        **kwargs,
    )
    return json.loads(res.choices[0].message.content), res.usage.prompt_tokens, res.usage.completion_tokens


def section_key(name):
    """답변이 표시한 항목 이름을 항목키로. 마감일·상태처럼 공고 정보를 가리키면 공고 개요로 본다."""
    if name in KOR_TO_SECTION:
        return KOR_TO_SECTION[name]
    if name in SECTION_KOR:
        return name
    return "overview" if any(w in name for w in ("마감", "개요", "상태")) else name


def score_sources(item, answer_sources):
    """근거 표시 정확도 (코드, 채점 기준 v2).
    - 정답 근거에 없는 공고를 표시하면 오답. 정답 문장에 회사명이 나오는 공고(A12 코즈코리아 등)는 허용
    - 정답 근거(공고ID+항목)를 모두 표시하면 정답
    - 공고는 맞지만 항목이 빠지거나 다르면 부분 (A10: 채용절차 대신 공고 개요를 표시한 경우)"""
    gold = set(gold_units(item["근거"]))
    if not gold:
        return ""
    gold_pids = {pid for pid, _ in gold}
    shown = {(s["posting_id"], section_key(s["section"])) for s in answer_sources}
    shown_pids = {pid for pid, _ in shown}
    allowed = gold_pids | {pid for pid in shown_pids if pid in POSTINGS and POSTINGS[pid]["company"] in item["정답"]}
    if shown_pids - allowed:
        return "오답"
    if gold <= shown:
        return "정답"
    return "부분" if shown_pids & gold_pids else "오답"


def grade(item, answer, sources, client, judge_model):
    """답변 정확도(LLM)와 근거 표시 정확도(코드). 답변 생성과 분리해 같은 답변을 다시 채점할 수 있게 한다."""
    judge_input = "\n".join([
        "[문항]",
        f"분류: {item['분류']} / 세부유형: {item['세부유형']}",
        f"질문: {item['질문']}",
        f"정답: {item['정답']}",
        f"필수 요소: {item['필수 요소']}",
        f"섞이면 안 되는 내용: {item['섞이면 안 되는 내용'] or '(없음)'}",
        f"채점 메모: {item['채점 메모'] or '(없음)'}",
        "",
        "[챗봇 답변]",
        answer,
    ])
    judge, j_in, j_out = chat_json(client, judge_model, JUDGE_SYSTEM, judge_input, JUDGE_SCHEMA, "judge", None)
    is_none_type = item["분류"] == "문서에 없음"
    return {
        "답변 판정": "" if is_none_type else judge["verdict"],
        "근거 표시 판정": "" if is_none_type else score_sources(item, sources),
        "문서에 없음 처리": judge["verdict"] if is_none_type else "",
        "채점자": f"LLM({judge_model}), 채점 기준 {GRADING_VERSION}",
        "LLM 판정 이유": judge["reason"]
                       + (f" | 빠짐: {', '.join(judge['missing'])}" if judge["missing"] else "")
                       + (f" | 섞임: {', '.join(judge['contaminated'])}" if judge["contaminated"] else ""),
    }, (j_in, j_out)


def fetch_full_posting(col, posting_id):
    """공고 하나의 청크 전부를 항목 순서(개요→…→채용절차), 항목 안에서는 청크 순서대로 가져온다."""
    res = col.get(where={"posting_id": posting_id})
    order = list(SECTION_KOR)
    pairs = sorted(zip(res["documents"], res["metadatas"]),
                   key=lambda p: (order.index(p[1]["section_key"]), p[1]["chunk_index_in_section"]))
    return [d for d, _ in pairs], [m for _, m in pairs]


FIELDS_CSV = ROOT / "data" / "posting_fields_eval.csv"
FOUR_YEAR = {"대학교졸업(4년) 이상", "석사졸업 이상"}

PARSE_SYSTEM = f"""너는 채용공고 검색 질문에서 공고를 거를 조건만 뽑는 해석기다. 오늘은 {REF_DATE}이다.
- 해당 없는 조건은 false, 빈 문자열, 빈 목록으로 둔다. 신입·직무 범위는 아래 newcomer_only, job으로만 뽑는다.
- education_not_4year: 4년제 대학 졸업이 필요 없는 곳을 찾는 질문
- closed_before: "N월 N일 전에 마감"이면 그 날짜(YYYY-MM-DD). 그 날짜는 포함하지 않는다
- remote_or_flexible: 재택·원격·하이브리드·유연근무·자율/시차 출퇴근이 되는 곳을 찾는 질문
- tech: 질문에 나온 기술 이름을 질문에 쓴 그대로 (예: 스프링, Node.js). 테스트 코드는 기술이 아니라 test_code로 둔다
- test_code: 테스트 코드 작성 경험을 우대·요구하는 곳을 찾는 질문
- coding_test: 코딩테스트나 과제 전형이 있는 곳을 찾는 질문
- newcomer_only: 질문이 신입 공고만 찾는 경우 true
- job: 질문이 한 직무의 공고만 찾는 경우 그 직무("백엔드" 또는 "프론트엔드"), 아니면 빈 문자열
- location: 근무지(지역)로 찾는 질문이면 지역명 목록. 시·도는 짧게(서울, 경기, 인천, 대전, 부산, 충남 등), 구·시는 "강남구", "성남시"처럼, 동네·역 이름은 질문에 쓴 그대로(판교, 가산). 재택·원격은 근무지가 아니라 remote_or_flexible로 둔다
- 공통 요구사항 요약, 비교, 연봉, 특정 회사 질문처럼 조건으로 공고를 거르는 질문이 아니면 모두 비워 둔다."""

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "education_not_4year": {"type": "boolean"},
        "closed_before": {"type": "string"},
        "remote_or_flexible": {"type": "boolean"},
        "tech": {"type": "array", "items": {"type": "string"}},
        "test_code": {"type": "boolean"},
        "coding_test": {"type": "boolean"},
        "location": {"type": "array", "items": {"type": "string"}},
        "newcomer_only": {"type": "boolean"},
        "job": {"type": "string"},
    },
    "required": ["education_not_4year", "closed_before", "remote_or_flexible", "tech", "test_code", "coding_test", "location",
                 "newcomer_only", "job"],
    "additionalProperties": False,
}

STRUCTURED_RULES = """

추가 규칙 (조건 코드 필터)
7. [코드 필터 결과]는 프로그램이 조건으로 거른 공고 목록이다. 이 목록에 있는 공고만 답하고, 목록에 없는 공고를 추가하지 않는다. 목록의 공고를 빼지도 않는다.
8. 각 공고가 조건에 맞는 이유를 [근거]에서 찾아 설명한다. 필수(자격요건)와 우대(우대사항)는 구분해서 말한다.
9. 목록이 비어 있으면 조건에 맞는 공고가 확인되지 않는다고 답한다."""

DIFF_RULES = """

추가 규칙 (분류-본문 차이)
10. [분류-본문 차이]가 있으면, 해당 공고에 대해 점핏 분류와 본문 내용이 다르다고 분명히 밝히고 어느 쪽을 근거로 말하는지 적는다. 점핏 분류(예: 신입)만 보고 지원에 문제없다고 단정하지 않는다.
11. [분류-본문 차이]의 '참고 공고'는 조건에 해당한다고 단정하지 않는다. 점핏 기술스택 태그에만 있고 본문 근거는 없다고만 말한다."""

# 답변 지시문 v2 (--prompt v2). v1은 위 상수를 그대로 둔다.
# 바뀐 것: 규칙 2에 일반 지식 추론 예시, 규칙 6을 항목 누락 방지로 교체, 규칙 7(지원 가능 여부는 결론부터) 추가,
# 조건 필터 규칙의 필수·우대 구분을 기술 조건 질문에만 적용. 규칙 번호는 이어서 다시 매겼다.
ANSWER_SYSTEM_V2 = f"""너는 점핏 개발자 채용공고를 근거로 답하는 상담 챗봇이다. 오늘은 {REF_DATE}이다.

규칙
1. 아래 [근거]에 있는 내용만으로 답한다. 근거에 없는 내용은 "확인되지 않습니다"라고 답하고, 추측하거나 외부 지식으로 채우지 않는다.
2. 질문을 잇기 위해 일반 지식(예: 프레임워크가 어떤 언어 기반인지)을 써야 하면, 그 부분이 공고 내용이 아니라 추론이라고 밝힌다. 예: "React는 JavaScript 라이브러리이므로 …(공고 내용이 아닌 일반 지식 추론)".
3. 공고를 언급할 때마다 회사명과 공고명을 쓰고, 마감일과 상태는 [근거]에 적힌 값을 그대로 옮긴다.
4. 마감된 공고는 지원할 수 없다고 분명히 말한다.
5. 서로 다른 공고의 내용을 섞지 않는다. 같은 회사라도 공고ID가 다르면 다른 공고다.
6. 질문에 대한 답만 한다. 다만 주요업무·자격요건·우대사항·전형·연봉·복지처럼 항목을 묻는 질문은 [근거]에 나열된 항목을 요약하거나 생략하지 말고 빠짐없이 쓴다. 불필요한 배경 설명은 붙이지 않는다.
7. 지원 가능 여부를 묻는 질문은 결론(가능 / 조건부 가능 / 불가 / 확인 불가)을 먼저 말하고, 근거의 필수 자격요건을 빠짐없이 함께 말한다.

출력 필드
- answer: 사용자에게 보여 줄 답변
- sources: 답변의 근거로 쓴 공고ID와 항목 (근거를 쓰지 않았으면 빈 목록)
- quotes: 답변의 근거가 된 원문 문장을 [근거]에서 글자 그대로 복사한 목록"""

STRUCTURED_RULES_V2 = """

추가 규칙 (조건 코드 필터)
8. [코드 필터 결과]는 프로그램이 조건으로 거른 공고 목록이다. 이 목록에 있는 공고만 답하고, 목록에 없는 공고를 추가하지 않는다. 목록의 공고를 빼지도 않는다.
9. 각 공고가 조건에 맞는 이유를 [근거]에서 찾아 설명한다. 기술을 조건으로 찾는 질문에서만 필수(자격요건)와 우대(우대사항)를 구분해서 말한다. 그 밖의 조건에서는 구분하지 않고, 근거 항목에 없다는 말을 덧붙이지 않는다.
10. 목록이 비어 있으면 조건에 맞는 공고가 확인되지 않는다고 답한다."""

DIFF_RULES_V2 = """

추가 규칙 (분류-본문 차이)
11. [분류-본문 차이]가 있으면, 해당 공고에 대해 점핏 분류와 본문 내용이 다르다고 분명히 밝히고 어느 쪽을 근거로 말하는지 적는다. 점핏 분류(예: 신입)만 보고 지원에 문제없다고 단정하지 않는다.
12. [분류-본문 차이]의 '참고 공고'는 조건에 해당한다고 단정하지 않는다. 점핏 기술스택 태그에만 있고 본문 근거는 없다고만 말한다."""

PROMPTS = {"v1": (ANSWER_SYSTEM, STRUCTURED_RULES, DIFF_RULES),
           "v2": (ANSWER_SYSTEM_V2, STRUCTURED_RULES_V2, DIFF_RULES_V2)}

SECTIONS_FOR = {"education_not_4year": {"overview"}, "closed_before": {"overview"},
                "remote_or_flexible": {"welfares", "qualifications"},
                "tech": {"qualifications", "preferredRequirements"},
                "test_code": {"qualifications", "preferredRequirements"}, "coding_test": {"recruitProcess"},
                "location": {"overview"}, "newcomer_only": {"overview"}, "job": {"overview"}}
COND_LABEL = {"education_not_4year": "4년제 졸업 불필요", "closed_before": "마감일이 {} 이전",
              "remote_or_flexible": "재택·유연근무 가능", "tech": "기술 {}", "test_code": "테스트 코드 언급",
              "coding_test": "코딩테스트·과제 전형", "location": "근무지 {}",
              "newcomer_only": "신입 공고", "job": "직무 {}"}
JOB_CATEGORY = {"백엔드": "서버/백엔드 개발자", "프론트엔드": "프론트엔드 개발자"}  # 직무 이름 → 점핏 직무 분류
# 이것만 있으면 공고를 거르는 질문이 아니다(범위만 말한 질문, 예: "티엔에이치 백엔드 신입은 …")
REAL_CONDITIONS = {"education_not_4year", "closed_before", "remote_or_flexible", "tech", "test_code", "coding_test", "location"}

# 근무지 표기 보정: 질문에서 뽑은 지역명을 공고 주소(예: "서울 강남구 …", "충남 아산시 …")의 표기로 맞춘다
LOCATION_ALIAS = {"서울시": "서울", "서울특별시": "서울", "경기도": "경기", "인천광역시": "인천", "대전광역시": "대전",
                  "부산광역시": "부산", "대구광역시": "대구", "광주광역시": "광주", "울산광역시": "울산",
                  "세종특별자치시": "세종", "충청남도": "충남", "충청북도": "충북", "전라남도": "전남", "전라북도": "전북",
                  "경상남도": "경남", "경상북도": "경북", "강원도": "강원", "제주도": "제주", "제주특별자치도": "제주"}


def load_fields():
    return {r["posting_id"]: r for r in load_csv(FIELDS_CSV)}


def canonical_tech(name, groups):
    for g in groups:
        if any(name.lower() == t.lower() for t in g):
            return g[0]
    return name


def active_conditions(parsed, groups):
    conds = {}
    for k, v in parsed.items():
        if k == "tech":
            if v:
                conds[k] = [canonical_tech(t, groups) for t in v]
        elif k == "location":
            if v:
                conds[k] = [LOCATION_ALIAS.get(x.strip(), x.strip()) for x in v if x.strip()]
        elif v:
            conds[k] = v
    return conds if set(conds) & REAL_CONDITIONS else {}


def apply_conditions(ids, conds, fields):
    """검색 범위 안에서 기준일에 진행 중인 공고만 남기고, 조건을 모두 만족하는 공고ID를 돌려준다."""
    out = []
    for pid in ids:
        p, f = POSTINGS[pid], fields[pid]
        if effective_status(p["closed_at"], p["status"], REF_DATE) != "진행 중":
            continue
        techs = {t.lower() for t in (f["tech_required"] + ";" + f["tech_preferred"]).split(";") if t}
        ok = all([
            not conds.get("education_not_4year") or p["education"] not in FOUR_YEAR,
            not conds.get("closed_before") or (p["closed_at"] != "상시" and p["closed_at"] < conds["closed_before"]),
            not conds.get("remote_or_flexible") or "Y" in (f["remote_work"], f["flexible_hours"]),
            all(t.lower() in techs for t in conds.get("tech", [])),
            not conds.get("test_code") or bool(f["test_code"]),
            not conds.get("coding_test") or f["coding_test"] == "Y",
            not conds.get("location") or any(x in p["location"] for x in conds["location"]),
            not conds.get("newcomer_only") or p["newcomer"] == "Y",
            not conds.get("job") or JOB_CATEGORY.get(conds["job"], conds["job"]) in p["job_categories"],
        ])
        if ok:
            out.append(pid)
    return out


def tag_only_refs(ids, conds, matched, fields, groups):
    """기술 조건에서 코드 필터(본문 기준)에는 안 걸렸지만 점핏 기술스택 태그에는 조건 기술이 모두 있는 공고."""
    if "tech" not in conds:
        return []
    refs = []
    for pid in ids:
        p = POSTINGS[pid]
        if pid in matched or effective_status(p["closed_at"], p["status"], REF_DATE) != "진행 중":
            continue
        tags = {canonical_tech(t.strip(), groups).lower() for t in p["tech_stacks"].replace(";", ",").split(",") if t.strip()}
        if all(t.lower() in tags for t in conds["tech"]):
            refs.append(pid)
    return refs


CAREER_ASK_RE = re.compile(r"신입|경력|연차|년차|지원해도|지원할 수|지원 가능|자격")


def build_diff_notes(pids, fields, conds=None, refs=(), career=True):
    """4단계: 점핏 분류와 본문이 다른 점을 답변 근거 앞에 붙일 문장으로 만든다. 없으면 빈 문자열.
    career=False면 경력 차이는 쓰지 않는다 (질문이 경력·신입 여부와 관련 없을 때)."""
    lines = []
    for pid in pids:
        f, p = fields[pid], POSTINGS[pid]
        if career and f.get("career_mismatch") == "Y":
            label = "신입" if p["newcomer"] == "Y" else f"경력 {p['career_min']}년 이상"
            lines.append(f"- 공고ID {pid} | {p['company']} · {p['title']}: 점핏 경력 분류는 {label}이지만 "
                         f"본문은 경력 {f['body_career_min']}년 이상 수준을 기대한다")
    if refs:
        techs = ", ".join(conds["tech"])
        lines.append(f"- 참고 공고({techs}): 점핏 기술스택 태그에는 있으나 본문 자격요건·우대사항에는 없어 코드 필터 결과에서 제외됨: "
                     + "; ".join(f"공고ID {pid} {POSTINGS[pid]['company']}" for pid in refs))
    return "[분류-본문 차이]\n" + "\n".join(lines) if lines else ""


def filter_summary(conds, matched, fields):
    label = ", ".join(COND_LABEL[k].format(", ".join(v) if isinstance(v, list) else v) for k, v in conds.items())
    lines = [f"조건: {label} (기준일 {REF_DATE} 진행 중인 공고만, 검색 범위 안에서) → 해당 공고 {len(matched)}건"]
    for pid in matched:
        p, f = POSTINGS[pid], fields[pid]
        lines.append(f"- 공고ID {pid} | {p['company']} · {p['title']} | 마감일 {p['closed_at']} | 근무지 {p['location']} | 학력 {p['education']} | "
                     f"재택 {f['remote_work']}·유연근무 {f['flexible_hours']} | 필수 기술 {f['tech_required'] or '-'} | "
                     f"우대 기술 {f['tech_preferred'] or '-'} | 테스트 코드 {f['test_code'] or '-'} | 코딩테스트 {f['coding_test']}")
    return "\n".join(lines)


def run_structured(item, ids, col, client, model, reasoning):
    """3단계: 질문을 조건으로 해석(LLM)하고 공고별 칸을 코드로 걸러낸다. 조건이 없으면 None(검색으로 처리)."""
    parsed, p_in, p_out = chat_json(client, model, PARSE_SYSTEM, item["질문"], PARSE_SCHEMA, "conditions", reasoning)
    groups = load_synonym_groups()
    conds = active_conditions(parsed, groups)
    if not conds:
        return None, (p_in, p_out)
    fields = load_fields()
    matched = apply_conditions(ids, conds, fields)
    refs = tag_only_refs(ids, conds, matched, fields, groups)
    keep = {"overview"}.union(*(SECTIONS_FOR[k] for k in conds))
    docs, metas = [], []
    for pid in matched:
        d, m = fetch_full_posting(col, pid)
        for doc, meta in zip(d, m):
            if meta["section_key"] in keep:
                docs.append(doc)
                metas.append(meta)
    return {"conds": conds, "matched": matched, "refs": refs, "docs": docs, "metas": metas,
            "text": filter_summary(conds, matched, fields)}, (p_in, p_out)


def per_posting_targets(item, ids, structured, structured_hit):
    """6단계: 공고별 균등 검색을 쓸 문항이면 대상 공고ID 목록, 아니면 None.
    - 조건 필터 문항인데 코드 필터로 해석되는 조건이 없으면(공통 요구처럼 요약을 묻는 질문) 검색 범위 안 공고 전부
    - 검색 범위가 전체이고 질문에 회사명이 2곳 이상 나오며 '신입'이 있으면 그 회사들의 신입 공고
      (같은 회사에 신입·경력 공고가 따로 있어 신입 조건 없이 회사명만으로 좁히면 경력 공고가 섞인다)"""
    if structured and item["검색 범위"].startswith("필터:") and not structured_hit:
        return ids
    if item["검색 범위"] == "전체" and "신입" in item["질문"]:
        companies = {p["company"] for p in POSTINGS.values() if p["company"] in item["질문"]}
        if len(companies) >= 2:
            return sorted(pid for pid, p in POSTINGS.items() if p["company"] in companies and p["newcomer"] == "Y")
    return None


# 질문의 낱말로 어느 항목을 묻는지 찾는 규칙 (앞에서부터 처음 맞는 항목). 6단계 공고별 검색용.
SECTION_HINTS = [
    ("preferredRequirements", ("우대",)),
    ("qualifications", ("자격요건", "자격", "요구", "필요한 것", "할 줄 알아야")),
    ("responsibility", ("주요업무", "하는 일", "무슨 일", "업무")),
    ("welfares", ("복지", "재택", "유연근무", "혜택")),
    ("recruitProcess", ("전형", "채용절차", "코딩테스트", "면접", "연봉")),
]


def target_section(question):
    for key, words in SECTION_HINTS:
        if any(w in question for w in words):
            return key
    return None


def query_per_posting(col, qvec, pids, k, question):
    """공고마다 근거를 가져와 공고 순서대로 붙인다.
    질문에서 묻는 항목을 찾으면 그 항목 청크 전부와 공고 개요(회사·직무·마감일)를, 못 찾으면 질문과 가까운 청크 k개를 가져온다."""
    section = target_section(question)
    docs, metas = [], []
    for pid in pids:
        if section:
            d, m = fetch_full_posting(col, pid)
            pairs = [(x, y) for x, y in zip(d, m) if y["section_key"] in ("overview", section)]
            docs += [x for x, _ in pairs]
            metas += [y for _, y in pairs]
        else:
            res = col.query(query_embeddings=[qvec], n_results=k, where={"posting_id": pid})
            docs += res["documents"][0]
            metas += res["metadatas"][0]
    return docs, metas


def run_one(item, col, qvec, top_k, client, model, judge_model, reasoning, full_posting=False, structured=False,
            diff_notes=False, diff_relevant_only=False, prompt="v1", per_posting=0):
    ids = scope_ids(item["검색 범위"])
    # 1단계: 공고를 지정한 문항은 검색 대신 그 공고 전체를 넣는다. 이때 검색 지표는 의미가 없어 비운다.
    full = full_posting and item["검색 범위"].startswith("공고 선택:")
    # 3단계: 조건으로 찾는 문항은 검색 대신 코드로 걸러 낸 공고만 넣는다. 조건이 없으면 기존 검색으로 처리한다.
    st, parse_tokens, pp_pids = None, None, None
    if structured and item["검색 범위"].startswith("필터:"):
        st, parse_tokens = run_structured(item, ids, col, client, model, reasoning)
    if st:
        docs, metas = st["docs"], st["metas"]
        retrieved, hit, recall, gold = [], "", "", []
    elif full:
        docs, metas = fetch_full_posting(col, ids[0])
        retrieved, hit, recall, gold = [], "", "", []
    else:
        pp_pids = per_posting_targets(item, ids, structured, st) if per_posting else None
        if pp_pids:
            docs, metas = query_per_posting(col, qvec, pp_pids, per_posting, item["질문"])
        else:
            where = {"posting_id": {"$in": ids}} if ids else None
            res = col.query(query_embeddings=[qvec], n_results=top_k, where=where)
            docs, metas = res["documents"][0], res["metadatas"][0]
        retrieved = [(m["posting_id"], m["section_key"]) for m in metas]

        gold = gold_units(item["근거"])
        found = [g for g in gold if g in retrieved]
        hit = ("O" if found else "X") if gold else ""
        recall = f"{len(found)}/{len(gold)}" if gold else ""

    context = build_context(list(zip(docs, metas)))
    # 4단계: 점핏 분류와 본문이 다른 공고가 근거에 있으면 그 차이를 근거 앞에 붙인다 (없으면 프롬프트는 그대로)
    notes = ""
    if diff_notes:
        pids = list(dict.fromkeys(m["posting_id"] for m in metas))
        asked = not diff_relevant_only or bool(CAREER_ASK_RE.search(item["질문"]))
        notes = build_diff_notes(pids, load_fields(), st["conds"] if st else None, st["refs"] if st else (), career=asked)
    base_rules, structured_rules, diff_rules = PROMPTS[prompt]
    blocks, system = [], base_rules
    if st:
        blocks.append(f"[코드 필터 결과]\n{st['text']}")
        system += structured_rules
    if notes:
        blocks.append(notes)
        system += diff_rules
    user = "\n\n".join(blocks + [f"[근거]\n{context}", f"[질문]\n{item['질문']}"])
    context = "\n\n".join(blocks + [context])  # 코드가 만든 필터 결과·차이 문장도 인용할 수 있게 검증 대상에 포함
    ans, a_in, a_out = chat_json(client, model, system, user, ANSWER_SCHEMA, "answer", reasoning)

    # 출처 정확도: 인용 문장이 LLM에 넘긴 근거(청크 본문 + 마감일·상태 줄)에 글자 그대로 있는지
    context_norm = normalize(context)
    quotes = [q for q in ans["quotes"] if q.strip()]
    verified = sum(normalize(q) in context_norm for q in quotes)

    graded, (j_in, j_out) = grade(item, ans["answer"], ans["sources"], client, judge_model)

    # 필터 정밀도·재현율: 코드가 거른 공고 vs 정답 근거에 나온 공고 (조건 필터를 쓴 문항만)
    f_ids, f_prec, f_rec, memo = "", "", "", ""
    if st:
        gold_pids = {pid for pid, _ in gold_units(item["근거"])}
        both = set(st["matched"]) & gold_pids
        f_ids = "; ".join(st["matched"])
        f_prec = f"{len(both)}/{len(st['matched'])}"
        f_rec = f"{len(both)}/{len(gold_pids)}" if gold_pids else ""
        memo = f"코드 필터 {json.dumps(st['conds'], ensure_ascii=False)} → {len(st['matched'])}건"
    elif full:
        memo = f"공고 전체 투입 {len(docs)}청크"
    if pp_pids:
        memo += ("; " if memo else "") + f"공고별 균등 검색 {len(pp_pids)}공고, 항목 {target_section(item['질문']) or '미지정(공고당 상위 ' + str(per_posting) + '개)'}"
    if notes:
        memo += ("; " if memo else "") + f"분류-본문 차이 표시 {notes.count(chr(10))}건"
    tokens = [(model, a_in, a_out), (judge_model, j_in, j_out)] + ([(model, *parse_tokens)] if parse_tokens else [])
    return {
        "id": item["id"],
        "분류": item["분류"],
        "질문": item["질문"],
        "검색 결과": "; ".join(f"{p}/{s}" for p, s in retrieved) if not (full or st) else "",
        "검색 적중": hit,
        "근거 재현율": recall,
        "답변": ans["answer"],
        "표시 근거": "; ".join(f"{s['posting_id']}/{s['section']}" for s in ans["sources"]),
        "인용": " | ".join(quotes),
        "인용 검증": f"{verified}/{len(quotes)}" if quotes else "0/0",
        **graded,
        "사람 판정": "",
        "메모": memo,
        "필터 결과": f_ids, "필터 정밀도": f_prec, "필터 재현율": f_rec,
        "개요 청크 수": sum(1 for _, s in retrieved if s == "overview"),
        "_tokens": tokens,
        "_quotes": (verified, len(quotes)),
    }


NO_CONTEXT_SYSTEM = f"""너는 개발자 채용 상담 챗봇이다. 오늘은 {REF_DATE}이다.
사용자는 채용 플랫폼 점핏에 올라온 개발자 채용공고에 대해 묻는다. 아는 범위에서 간결하게 답한다.

출력 필드
- answer: 사용자에게 보여 줄 답변"""

NO_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def run_no_context(item, client, model, judge_model, reasoning):
    """GPT 빈손 테스트: 공고 문서 없이 같은 질문에 답하게 하고 같은 기준으로 채점한다.
    공고를 고른 문항은 화면에서 보이는 정보(회사명·공고명)만 질문 앞에 붙인다."""
    ids = scope_ids(item["검색 범위"])
    question = item["질문"]
    if item["검색 범위"].startswith("공고 선택:"):
        p = POSTINGS[ids[0]]
        question = f"(선택한 공고: {p['company']} · {p['title']})\n{question}"
    elif item["검색 범위"].startswith("필터:"):
        question = f"({item['검색 범위']})\n{question}"
    ans, a_in, a_out = chat_json(client, model, NO_CONTEXT_SYSTEM, question, NO_CONTEXT_SCHEMA, "answer", reasoning)
    graded, (j_in, j_out) = grade(item, ans["answer"], [], client, judge_model)
    return {
        "id": item["id"], "분류": item["분류"], "질문": question,
        "검색 결과": "", "검색 적중": "", "근거 재현율": "",
        "답변": ans["answer"], "표시 근거": "", "인용": "", "인용 검증": "0/0",
        **graded, "근거 표시 판정": "",
        "사람 판정": "", "메모": "", "필터 결과": "", "필터 정밀도": "", "필터 재현율": "", "개요 청크 수": 0,
        "_tokens": [(model, a_in, a_out), (judge_model, j_in, j_out)],
        "_quotes": (0, 0),
    }


def eval_set_tag():
    """기본 평가셋(eval_set.csv)이 아니면 결과 파일·설정에 붙일 평가셋 이름. 개발 문항 결과 파일을 덮어쓰지 않게 한다."""
    return "" if EVAL_SET.name == "eval_set.csv" else f"_{EVAL_SET.stem}"


def summarize(rows):
    def rate(vals):
        return f"{sum(vals) / len(vals):.1%}" if vals else ""
    retr = [r for r in rows if r["검색 적중"]]
    answerable = [r for r in rows if r["답변 판정"]]
    shown = [r for r in rows if r["근거 표시 판정"]]
    none_rows = [r for r in rows if r["문서에 없음 처리"]]
    fresh = [r for r in rows if r["id"] in FRESHNESS_IDS]
    q_ok = sum(r["_quotes"][0] for r in rows)
    q_all = sum(r["_quotes"][1] for r in rows)
    filt = [r for r in rows if r.get("필터 정밀도")]
    prec_vals = [int(a) / int(b) for a, b in (r["필터 정밀도"].split("/") for r in filt) if int(b)]
    frec_vals = [int(a) / int(b) for a, b in (r["필터 재현율"].split("/") for r in filt if r["필터 재현율"]) if int(b)]
    recall_vals = [int(a) / int(b) for a, b in (r["근거 재현율"].split("/") for r in retr)]
    return {
        "최신성": rate([VERDICT_SCORE[r["답변 판정"]] for r in fresh]),
        "출처 정확도": f"{q_ok / q_all:.1%} ({q_ok}/{q_all})" if q_all else "",
        "답변 정확도": rate([VERDICT_SCORE[r["답변 판정"]] for r in answerable]),
        "검색 적중률": rate([r["검색 적중"] == "O" for r in retr]),
        "근거 재현율": rate(recall_vals),
        "필터 정밀도": rate(prec_vals),
        "필터 재현율": rate(frec_vals),
        "근거 표시 정확도": rate([VERDICT_SCORE[r["근거 표시 판정"]] for r in shown]),
        "문서에 없음 처리": rate([r["문서에 없음 처리"] == "정답" for r in none_rows]),
        "개요 청크 비율": rate([r["개요 청크 수"] / len(r["검색 결과"].split("; ")) for r in rows if r["검색 결과"]]),
        "사람 일치율": rate([r["사람 판정"] == (r["답변 판정"] or r["문서에 없음 처리"])
                          for r in rows if r["사람 판정"]]),
    }


def write_results(rows, out, setting):
    """전체 채점표는 raw/에, 공고 원문 인용 열을 뺀 공개 채점표는 results/에 저장한다."""
    cols = [k for k in rows[0] if not k.startswith("_") and k != "설정"]
    for path, keep in ((RAW_RESULTS_DIR / out.name, cols),
                       (out, [c for c in cols if c not in PRIVATE_COLUMNS])):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["설정"] + keep)
            w.writeheader()
            for r in rows:
                w.writerow({"설정": setting, **{k: r.get(k, "") for k in keep}})


def append_summary(row):
    """지표가 늘어나면 기존 줄은 빈칸으로 두고 헤더를 넓힌다."""
    old = load_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else []
    fields = list(old[0].keys()) if old else []
    fields += [k for k in row if k not in fields]
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old + [row])


def report(rows, out, setting, record):
    summary = summarize(rows)
    print(f"\n[{setting}] {len(rows)}문항 → {out.relative_to(ROOT).as_posix()}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if record:
        append_summary({"시각": datetime.now().isoformat(timespec="seconds"), "설정": setting,
                        "결과 파일": out.name, **summary})


def regrade(raw_name, client, judge_model, workers):
    """같은 답변을 현재 평가셋·채점 기준으로 다시 채점한다 (답변 생성 없이 채점 기준 변경 효과만 비교)."""
    items = {it["id"]: it for it in load_csv(EVAL_SET)}
    old_rows = load_csv(RAW_RESULTS_DIR / raw_name)

    def one(r):
        sources = []
        for x in r["표시 근거"].split("; "):
            if not x:
                continue
            if "/" in x:
                pid, sec = x.split("/", 1)
                sources.append({"posting_id": pid, "section": sec})
            elif sources:  # 항목 이름에 "; "가 들어 있던 경우: 앞 출처의 항목 이름에 이어 붙인다
                sources[-1]["section"] += "; " + x
        graded, tokens = grade(items[r["id"]], r["답변"], sources, client, judge_model)
        ok, total = (int(v) for v in r["인용 검증"].split("/"))
        return {**{k: v for k, v in r.items() if k != "설정"}, **graded,
                "개요 청크 수": int(r["개요 청크 수"]), "_tokens": [(judge_model, *tokens)], "_quotes": (ok, total)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, old_rows))
    setting = f"{old_rows[0]['설정']}, 재채점(답변 재사용) 채점 기준 {GRADING_VERSION}"
    out = RESULTS_DIR / raw_name.replace(".csv", f"_grade-{GRADING_VERSION}.csv")
    return rows, out, setting


def main():
    global EVAL_SET, FRESHNESS_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/eval_set.csv",
                    help="평가셋 CSV (프로젝트 폴더 기준). 기본값이 아니면 결과 파일 이름에 평가셋 이름이 붙는다. "
                         "최신성 문항은 평가셋의 '최신성' 열이 Y인 문항 (열이 없으면 개발 문항의 A19·A20·T04)")
    ap.add_argument("--strategy", nargs="+", default=["A", "B"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--judge-model", default="gpt-6-luna")
    ap.add_argument("--reasoning", default=None, help="reasoning_effort (기본: 모델 기본값)")
    ap.add_argument("--ids", nargs="*", help="일부 문항만 실행 (시험용)")
    ap.add_argument("--regrade", nargs="*", help="eval/results/raw/의 채점표 파일명. 답변은 그대로 두고 다시 채점")
    ap.add_argument("--full-posting", action="store_true", help="1단계: '공고 선택' 문항은 검색 대신 그 공고 전체를 근거로 투입")
    ap.add_argument("--synonyms", action="store_true", help="2단계: 질문의 기술 표기에 동의어를 덧붙여 검색 (data/tech_synonyms.json)")
    ap.add_argument("--structured", action="store_true", help="3단계: 조건으로 찾는 문항은 공고별 칸(data/posting_fields_eval.csv)을 코드로 걸러 LLM은 설명만")
    ap.add_argument("--diff-notes", action="store_true", help="4단계: 점핏 분류와 본문이 다른 공고는 그 차이를 근거에 붙여 답변에 명시 (경력은 data/posting_fields_eval.csv의 career_mismatch)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1", help="5단계: 답변 지시문 버전 (v2: 항목 누락 방지, 지원 가능 여부는 결론부터, 필수·우대 구분은 기술 조건만)")
    ap.add_argument("--per-posting", type=int, default=0, help="6단계: 공고별 균등 검색. 값 = 공고당 가져올 청크 수 (0이면 끔). 조건 없는 필터 질문은 범위 안 공고 전부, 회사 2곳 이상 비교 질문은 그 회사의 신입 공고")
    ap.add_argument("--label", default="", help="같은 설정을 여러 번 실행할 때 결과 파일 이름 끝에 붙일 라벨 (예: a, b)")
    ap.add_argument("--diff-relevant-only", action="store_true", help="--diff-notes와 함께: 경력 차이는 질문이 경력·신입 여부를 물을 때만 붙임")
    ap.add_argument("--no-context", action="store_true", help="GPT 빈손 테스트: 공고 문서 없이 답변")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    EVAL_SET = ROOT / args.eval_set
    if EVAL_SET.exists():
        FRESHNESS_IDS = {it["id"] for it in load_csv(EVAL_SET) if it.get("최신성") == "Y"} or FRESHNESS_IDS

    load_dotenv(ROOT / ".env")
    client = OpenAI()
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    usage = {}

    def add_usage(rows):
        for r in rows:
            for m, i, o in r["_tokens"]:  # 답변·채점 모델이 같아도 합산되도록 목록으로 받는다
                usage.setdefault(m, [0, 0])
                usage[m][0] += i
                usage[m][1] += o

    if args.no_context:
        items = load_csv(EVAL_SET)
        if args.ids:
            items = [it for it in items if it["id"] in set(args.ids)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda it: run_no_context(it, client, args.model, args.judge_model, args.reasoning),
                                 items))
        add_usage(rows)
        setting = (f"문서 없음(GPT 빈손), 답변 {args.model}(reasoning {args.reasoning or '기본'}), "
                   f"채점 {args.judge_model}, 채점 기준 {GRADING_VERSION}" + (f", 평가셋 {EVAL_SET.stem}" if eval_set_tag() else ""))
        tag = eval_set_tag() + ("" if not args.ids else "_partial")
        out = RESULTS_DIR / f"{date.today().isoformat()}_no-context_{args.model}{tag}.csv"
        write_results(rows, out, setting)
        report(rows, out, setting, record=not args.ids)
        memo, n_items = "GPT 빈손 테스트", len(items)
    elif args.regrade:
        for name in args.regrade:
            rows, out, setting = regrade(name, client, args.judge_model, args.workers)
            add_usage(rows)
            write_results(rows, out, setting)
            report(rows, out, setting, record=True)
        memo = f"재채점 {', '.join(args.regrade)}"
        n_items = sum(len(load_csv(RAW_RESULTS_DIR / n)) for n in args.regrade)
    else:
        items = load_csv(EVAL_SET)
        if args.ids:
            items = [it for it in items if it["id"] in set(args.ids)]
        groups = load_synonym_groups() if args.synonyms else []
        expanded = [expand_query(it["질문"], groups) if args.synonyms else (it["질문"], []) for it in items]
        emb = client.embeddings.create(model=EMBED_MODEL, input=[q for q, _ in expanded])
        qvecs = [d.embedding for d in sorted(emb.data, key=lambda d: d.index)]
        usage[EMBED_MODEL] = [emb.usage.total_tokens, 0]

        db = chromadb.PersistentClient(path=str(CHROMA_DIR))
        for strategy in args.strategy:
            col = db.get_collection(collection_name(strategy))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(
                    lambda pair: run_one(pair[0], col, pair[1], args.top_k, client,
                                         args.model, args.judge_model, args.reasoning, args.full_posting, args.structured,
                                         args.diff_notes, args.diff_relevant_only, args.prompt, args.per_posting),
                    zip(items, qvecs)))
            for r, (_, extra) in zip(rows, expanded):
                if extra and not r["메모"].startswith(("공고 전체", "코드 필터", "분류-본문")):
                    r["메모"] = (r["메모"] + "; " if r["메모"] else "") + f"검색 질문에 덧붙임: {', '.join(extra)}"
            add_usage(rows)
            setting = (f"전략 {strategy}, top-{args.top_k}, 임베딩 {EMBED_MODEL}, 답변 {args.model}"
                       f"(reasoning {args.reasoning or '기본'}), 채점 {args.judge_model}, "
                       f"프롬프트 {args.prompt}, 채점 기준 {GRADING_VERSION}"
                       + (f", 실행 라벨 {args.label}" if args.label else "")
                       + (f", 평가셋 {EVAL_SET.stem}" if eval_set_tag() else "")
                       + (f", 공고별 균등 검색(질문의 항목 청크 + 개요, 항목을 못 찾으면 공고당 상위 {args.per_posting}개)" if args.per_posting else "")
                       + (", 공고 선택 시 공고 전체 투입" if args.full_posting else "")
                       + (", 기술 동의어 확장" if args.synonyms else "")
                       + (", 조건 코드 필터" if args.structured else "")
                       + ((", 분류-본문 차이 표시(경력 차이는 질문이 경력·신입을 물을 때만)" if args.diff_relevant_only
                           else ", 분류-본문 차이 표시") if args.diff_notes else ""))
            tag = ("_full" if args.full_posting else "") + ("_syn" if args.synonyms else "") + ("_struct" if args.structured else "") + (("_diffrel" if args.diff_relevant_only else "_diff") if args.diff_notes else "") + (f"_pp{args.per_posting}" if args.per_posting else "") + ("_p2" if args.prompt == "v2" else "") + (f"_{args.label}" if args.label else "") + eval_set_tag() + ("" if not args.ids else "_partial")
            out = RESULTS_DIR / f"{date.today().isoformat()}_{strategy}_k{args.top_k}_{args.model}{tag}.csv"
            write_results(rows, out, setting)
            report(rows, out, setting, record=not args.ids)
        memo = f"전략 {'/'.join(args.strategy)}, top-{args.top_k}"
        n_items = len(items) * len(args.strategy)

    total = 0.0
    for m, (i, o) in usage.items():
        price_in, price_out = (EMBED_PRICE_PER_1M, 0) if m == EMBED_MODEL else PRICES[m]
        cost = i / 1e6 * price_in + o / 1e6 * price_out
        total += cost
        log_usage({"시각": datetime.now().isoformat(timespec="seconds"), "작업": "run_eval", "모델": m,
                   "입력 토큰": i, "출력 토큰": o, "비용(달러)": f"{cost:.5f}",
                   "메모": f"{memo}, {n_items}문항"})
    print(f"\n이번 실행 비용 약 ${total:.4f} (eval/usage_log.csv 기록)")


if __name__ == "__main__":
    main()
