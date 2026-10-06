"""6단계 그래프 생성 (SVG, 외부 패키지 없음).

결과 CSV(eval/results/)에서 숫자를 읽어 docs/figures/에 그림 3개를 만든다.
1. stage_scores.svg     : 단계별 답변 정확도와 최신성 (5·6단계는 2회 실행의 평균과 범위)
2. item_heatmap.svg     : 오답노트 13문항의 단계별 판정
3. top_k_retrieval.svg  : top_k별 검색 적중률 (eval/results/topk_retrieval.csv)

사용법
    .venv\\Scripts\\python.exe src\\make_figures.py
"""

import csv
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "figures"

P = "2026-10-06_B_k5_gpt-6-luna_"
# (단계 번호, 단계 이름, 결과 파일 목록(2회 실행이면 둘))
STAGES = [
    (0, "기준선", ["2026-10-02_B_k5_gpt-6-luna_grade-v2.csv"]),
    (1, "공고 전체 투입", [P + "full.csv"]),
    (2, "기술 동의어", [P + "full_syn.csv"]),
    (3, "조건 코드 필터", [P + "full_syn_struct.csv"]),
    (4, "분류-본문 차이", [P + "full_syn_struct_diffrel.csv"]),
    (5, "지시문 v2", [P + "full_syn_struct_diffrel_p2_a.csv", P + "full_syn_struct_diffrel_p2_b.csv"]),
    (6, "공고별 균등 검색", [P + "full_syn_struct_diffrel_pp2_p2_a.csv", P + "full_syn_struct_diffrel_pp2_p2_b.csv"]),
]
SCORE = {"정답": 1.0, "부분": 0.5, "오답": 0.0}
FRESH_IDS = {"A19", "A20", "T04"}
SHORT = {0: "기준선", 1: "전체 투입", 2: "동의어", 3: "코드 필터", 4: "분류-본문", 5: "지시문 v2", 6: "균등 검색"}  # 히트맵 열 제목
FONT = "'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, GREY = "#2563eb", "#9ca3af"
CELL = {"정답": "#8cc9a1", "부분": "#f3d37a", "오답": "#e8918a"}
SYM = {"정답": "O", "부분": "△", "오답": "X"}


def load(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def verdict(row):
    return row["답변 판정"] or row["문서에 없음 처리"]


def accuracy(rows, human=False):
    """정답이 있는 문항(답변 판정이 있는 행)의 평균 점수(%). human이면 사람 판정이 있는 문항은 그 값을 쓴다."""
    vals = []
    for r in rows.values():
        if not r["답변 판정"]:
            continue
        v = r["사람 판정"] if human and r["사람 판정"] in SCORE else r["답변 판정"]
        vals.append(SCORE[v])
    return sum(vals) / len(vals) * 100


def freshness(rows):
    return sum(SCORE[rows[i]["답변 판정"]] for i in FRESH_IDS) / len(FRESH_IDS) * 100


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{escape(s)}</text>")


def svg(width, height, title, body, desc):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
            f"role='img' aria-label='{escape(title)}' font-family=\"{FONT}\">"
            f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>"
            f"<rect width='{width}' height='{height}' fill='white'/>{body}</svg>")


def fig_stage_scores():
    stats = []
    for n, name, files in STAGES:
        runs = [load(f) for f in files]
        acc = [accuracy(r) for r in runs]
        hum = [accuracy(r, human=True) for r in runs]
        stats.append({"n": n, "name": name, "acc": sum(acc) / len(acc), "lo": min(acc), "hi": max(acc),
                      "hum": sum(hum) / len(hum), "runs": len(runs), "fresh": sum(freshness(r) for r in runs) / len(runs)})
    W, H, x0, step, ytop, ybot, lo, hi = 900, 470, 90, 112, 110, 350, 40, 100
    X = lambda i: x0 + i * step
    Y = lambda v: ybot - (v - lo) / (hi - lo) * (ybot - ytop)
    b = [text(40, 34, "답변 정확도는 3단계까지 66%에서 82%로 오르고, 최신성은 3단계에서 100%가 됐다", 18, INK, weight="bold")]
    # 범례
    b.append(f"<line x1='40' x2='64' y1='64' y2='64' stroke='{BLUE}' stroke-width='3'/><circle cx='52' cy='64' r='5' fill='{BLUE}'/>")
    b.append(text(72, 69, "답변 정확도(LLM 채점)", 13, MUTED))
    b.append(f"<circle cx='262' cy='64' r='6' fill='white' stroke='{BLUE}' stroke-width='2'/>")
    b.append(text(274, 69, "사람 판정 반영", 13, MUTED))
    b.append(f"<line x1='392' x2='416' y1='64' y2='64' stroke='{GREY}' stroke-width='3'/><circle cx='404' cy='64' r='4' fill='{GREY}'/>")
    b.append(text(424, 69, "최신성", 13, MUTED))
    b.append(f"<line x1='500' x2='500' y1='56' y2='72' stroke='{BLUE}' stroke-width='1.5'/>"
             f"<line x1='495' x2='505' y1='56' y2='56' stroke='{BLUE}'/><line x1='495' x2='505' y1='72' y2='72' stroke='{BLUE}'/>")
    b.append(text(512, 69, "같은 설정 2회의 범위(5·6단계)", 13, MUTED))
    b.append(f"<line x1='{x0 - 30}' x2='{X(6) + 30}' y1='{ybot}' y2='{ybot}' stroke='#9ca3af'/>")
    # 최신성
    pts = " ".join(f"{X(s['n'])},{Y(s['fresh'])}" for s in stats)
    b.append(f"<polyline points='{pts}' fill='none' stroke='{GREY}' stroke-width='2.5'/>")
    for s in stats:
        b.append(f"<circle cx='{X(s['n'])}' cy='{Y(s['fresh'])}' r='4' fill='{GREY}'><title>{s['n']} {escape(s['name'])} · 최신성 {s['fresh']:.1f}%</title></circle>")
    for n in (0, 3):
        s = stats[n]
        b.append(text(X(n), Y(s["fresh"]) - 12, f"{s['fresh']:.1f}%", 13, MUTED, "middle"))
    # 답변 정확도
    pts = " ".join(f"{X(s['n'])},{Y(s['acc'])}" for s in stats)
    b.append(f"<polyline points='{pts}' fill='none' stroke='{BLUE}' stroke-width='3'/>")
    for s in stats:
        x = X(s["n"])
        if s["runs"] > 1:
            b.append(f"<line x1='{x}' x2='{x}' y1='{Y(s['lo'])}' y2='{Y(s['hi'])}' stroke='{BLUE}' stroke-width='1.5'/>"
                     f"<line x1='{x - 5}' x2='{x + 5}' y1='{Y(s['lo'])}' y2='{Y(s['lo'])}' stroke='{BLUE}'/>"
                     f"<line x1='{x - 5}' x2='{x + 5}' y1='{Y(s['hi'])}' y2='{Y(s['hi'])}' stroke='{BLUE}'/>")
        tip = f"{s['n']} {s['name']} · 답변 정확도 {s['acc']:.1f}%" + (f" (두 번 {s['lo']:.1f}~{s['hi']:.1f}%)" if s["runs"] > 1 else "")
        b.append(f"<circle cx='{x}' cy='{Y(s['acc'])}' r='5' fill='{BLUE}'><title>{escape(tip)}</title></circle>")
        label_y = Y(s["lo"]) + 22 if s["runs"] > 1 else Y(s["acc"]) + 22
        b.append(text(x, label_y, f"{s['acc']:.1f}%", 13, INK, "middle", "bold"))
        if s["hum"] > s["acc"] + 0.01:
            b.append(f"<circle cx='{x}' cy='{Y(s['hum'])}' r='6' fill='white' stroke='{BLUE}' stroke-width='2'><title>{s['n']} {escape(s['name'])} · 사람 판정 반영 {s['hum']:.1f}%</title></circle>")
            b.append(text(x + 12, Y(s["hum"]) + 4, f"{s['hum']:.1f}%", 13, INK))
    for s in stats:
        x = X(s["n"])
        b.append(text(x, ybot + 24, str(s["n"]), 14, INK, "middle", "bold"))
        b.append(text(x, ybot + 42, s["name"], 12, MUTED, "middle"))
    b.append(text(40, H - 28, "전략 B, 채점 기준 v2. 세로축은 40%부터 시작한다. 1~4단계는 1회, 5·6단계는 같은 설정 2회 실행의 평균과 범위.", 12, MUTED))
    b.append(text(40, H - 10, "사람 판정 반영은 A14(채점 LLM의 반복 오판)를 사람이 판정한 값이다. 문항 하나가 2~4%p를 움직인다.", 12, MUTED))
    return svg(W, H, "단계별 답변 정확도와 최신성", "".join(b),
               "전략 B의 0~6단계 답변 정확도(LLM 채점, 사람 판정 반영)와 최신성")


def fig_heatmap():
    groups = [
        ("여러 공고를 모아야 함", ["A11", "A16", "A20"]),
        ("공고 안에서도 놓침", ["A10", "T03"]),
        ("한글 기술 이름", ["T01"]),
        ("일부만 찾음", ["A14", "A17"]),
        ("답을 덜 함", ["A03", "A04", "A09"]),
        ("중요한 조건을 안 말함", ["T02"]),
        ("추론을 밝히지 않음", ["A05"]),
    ]
    cols = []  # (헤더 1줄, 헤더 2줄, 파일)
    for n, name, files in STAGES:
        for k, f in enumerate(files):
            label = str(n) if len(files) == 1 else f"{n}-{'ab'[k]}"
            cols.append((label, SHORT[n] if k == 0 else "", f))
    data = [load(f) for _, _, f in cols]
    cw, rh, left, top = 62, 30, 250, 118
    n_rows = sum(len(g[1]) for g in groups)
    W, H = left + cw * len(cols) + 40, top + rh * n_rows + 90
    b = [text(40, 34, "검색 문제로 틀리던 문항은 풀렸고, 답변 누락·추론 문항(A03·A04·A05·A09)과 A16이 남았다", 18, INK, weight="bold")]
    b.append(text(40, 58, "O 정답 · △ 부분 · X 오답 (LLM 채점, 채점 기준 v2)", 13, MUTED))
    for j, (label, name, _) in enumerate(cols):
        x = left + j * cw + cw / 2
        b.append(text(x, top - 30, label, 14, INK, "middle", "bold"))
        if name:
            b.append(text(x, top - 12, name, 11, MUTED, "middle"))
    y = top
    for gi, (gname, ids) in enumerate(groups):
        if gi:
            b.append(f"<line x1='40' x2='{left + cw * len(cols)}' y1='{y}' y2='{y}' stroke='{GRID}'/>")
        for ri, i in enumerate(ids):
            if ri == 0:
                b.append(text(40, y + 20, gname, 12, MUTED))
            b.append(text(left - 14, y + 20, i, 14, INK, "end", "bold"))
            for j, rows in enumerate(data):
                v = verdict(rows[i])
                b.append(f"<rect x='{left + j * cw + 2}' y='{y + 2}' width='{cw - 4}' height='{rh - 4}' rx='4' fill='{CELL[v]}'>"
                         f"<title>{i} · {cols[j][0]}단계: {v}</title></rect>")
                b.append(text(left + j * cw + cw / 2, y + 21, SYM[v], 14, INK, "middle"))
            y += rh
    b.append(text(40, H - 36, "1~4단계는 1회 실행, 5·6단계는 같은 설정 2회(a, b). 1~4단계 열 아래 이름은 그 단계에서 추가한 것이다.", 12, MUTED))
    b.append(text(40, H - 18, "5·6단계의 a·b가 다른 문항은 노이즈로 본다. A14의 X에는 채점 LLM의 반복 오판이 섞여 있다(사람 판정은 정답 또는 부분).", 12, MUTED))
    return svg(W, H, "오답노트 13문항의 단계별 판정", "".join(b), "오답노트의 13문항이 0~6단계에서 정답·부분·오답 중 무엇이었는지")


def fig_topk():
    with (RESULTS / "topk_retrieval.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ks = sorted({int(r["top_k"]) for r in rows})
    series = {m: {int(r["top_k"]): float(r["검색 적중률(%)"]) for r in rows if r["설정"] == m} for m in ("질문 그대로", "동의어 확장")}
    W, H, x0, step, ytop, ybot, lo, hi = 760, 400, 110, 170, 100, 300, 50, 100
    X = lambda k: x0 + ks.index(k) * step
    Y = lambda v: ybot - (v - lo) / (hi - lo) * (ybot - ytop)
    b = [text(40, 34, "top_k를 5에서 10으로 올리면 순수 검색 적중률이 76%에서 92%로 오른다", 18, INK, weight="bold")]
    b.append(f"<line x1='40' x2='64' y1='64' y2='64' stroke='{BLUE}' stroke-width='3'/><circle cx='52' cy='64' r='5' fill='{BLUE}'/>")
    b.append(text(72, 69, "질문 그대로 (1차 기준선 설정)", 13, MUTED))
    b.append(f"<line x1='310' x2='334' y1='64' y2='64' stroke='{GREY}' stroke-width='3'/><circle cx='322' cy='64' r='4' fill='{GREY}'/>")
    b.append(text(342, 69, "동의어 확장 (2단계 설정)", 13, MUTED))
    b.append(f"<line x1='{x0 - 40}' x2='{X(ks[-1]) + 40}' y1='{ybot}' y2='{ybot}' stroke='#9ca3af'/>")
    b.append(f"<line x1='{X(5)}' x2='{X(5)}' y1='{ytop - 10}' y2='{ybot}' stroke='{GRID}' stroke-dasharray='4 4'/>")
    b.append(text(X(5) + 6, ytop - 2, "현재 기본값 k=5", 12, MUTED))
    for m, color, w, rr in (("동의어 확장", GREY, 2.5, 4), ("질문 그대로", BLUE, 3, 5)):
        pts = " ".join(f"{X(k)},{Y(series[m][k])}" for k in ks)
        b.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{w}'/>")
        for k in ks:
            b.append(f"<circle cx='{X(k)}' cy='{Y(series[m][k])}' r='{rr}' fill='{color}'><title>{m} · k={k}: {series[m][k]:.1f}%</title></circle>")
    for k in ks:  # 질문 그대로 값을 아래에, 동의어 확장은 k=5에서만 위에 따로 표시
        b.append(text(X(k), Y(series["질문 그대로"][k]) + 22, f"{series['질문 그대로'][k]:.0f}%", 13, INK, "middle", "bold"))
        b.append(text(X(k), ybot + 24, f"k={k}", 13, INK, "middle"))
    b.append(text(X(5), Y(series["동의어 확장"][5]) - 12, f"{series['동의어 확장'][5]:.0f}% (동의어)", 12, MUTED, "middle"))
    b.append(text(40, H - 28, "전략 B, 정답 근거가 있는 25문항. 질문을 임베딩해 정답 근거 청크가 상위 k개에 하나라도 있으면 적중. 세로축은 50%부터.", 12, MUTED))
    b.append(text(40, H - 10, "같은 설정으로 답변까지 평가하면(2회) 답변 정확도 66.0% → 78.0% / 68.0% (평균 73.0%). 파이프라인 전체에서는 차이가 없다.", 12, MUTED))
    return svg(W, H, "top_k별 검색 적중률", "".join(b), "top_k 3, 5, 10, 20에서 질문 그대로와 동의어 확장의 검색 적중률")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("stage_scores.svg", fig_stage_scores), ("item_heatmap.svg", fig_heatmap),
                     ("top_k_retrieval.svg", fig_topk)):
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("→", (OUT / name).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
