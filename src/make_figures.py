"""6단계 그래프 생성 (SVG, 외부 패키지 없음).

결과 CSV(eval/results/)에서 숫자를 읽어 docs/figures/에 그림 3개를 만든다.
정답지 수정 뒤(채점 기준 v4)를 기본으로 하고, stage_scores는 수정 전(v2)을 같이 그린다.
1. stage_scores.svg     : 단계별 답변 정확도 v2 → v4와 최신성 (5·6단계·top_k=10은 2회 실행의 평균과 각 값)
2. item_heatmap.svg     : 오답노트 13문항의 단계별 판정 (v4)
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
P10 = "2026-10-06_B_k10_gpt-6-luna_"
# (단계 번호, 단계 이름, 채점 기준 v2 결과 파일 목록(2회 실행이면 둘))
STAGES_V2 = [
    (0, "기준선", ["2026-10-02_B_k5_gpt-6-luna_grade-v2.csv"]),
    (1, "공고 전체 투입", [P + "full.csv"]),
    (2, "기술 동의어", [P + "full_syn.csv"]),
    (3, "조건 코드 필터", [P + "full_syn_struct.csv"]),
    (4, "분류-본문 차이", [P + "full_syn_struct_diffrel.csv"]),
    (5, "지시문 v2", [P + "full_syn_struct_diffrel_p2_a.csv", P + "full_syn_struct_diffrel_p2_b.csv"]),
    (6, "공고별 균등 검색", [P + "full_syn_struct_diffrel_pp2_p2_a.csv", P + "full_syn_struct_diffrel_pp2_p2_b.csv"]),
    (7, "top_k=10 최종", [P10 + "full_syn_struct_diffrel_pp2_p2_a.csv", P10 + "full_syn_struct_diffrel_pp2_p2_b.csv"]),
]
# 같은 답변을 정답지 수정(v4) 뒤 다시 채점한 파일. 기준선만 이름 규칙이 다르다.
STAGES = [(n, name, [f.replace("_grade-v2.csv", ".csv").replace(".csv", "_grade-v4.csv") for f in files])
          for n, name, files in STAGES_V2]
FIXED_IDS = {"A03", "A04", "A09", "A14"}  # v4에서 정답지를 고친 문항
SCORE = {"정답": 1.0, "부분": 0.5, "오답": 0.0}
FRESH_IDS = {"A19", "A20", "T04"}
SHORT = {0: "기준선", 1: "전체 투입", 2: "동의어", 3: "코드 필터", 4: "분류-본문", 5: "지시문 v2", 6: "균등 검색"}  # 히트맵 열 제목
V2_GREY = "#9ca3af"
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


def stage_stats(stages):
    out = []
    for n, name, files in stages:
        runs = [load(f) for f in files]
        acc = [accuracy(r) for r in runs]
        out.append({"n": n, "name": name, "accs": acc, "acc": sum(acc) / len(acc), "runs": len(runs),
                    "fresh": sum(freshness(r) for r in runs) / len(runs)})
    return out


def fig_stage_scores():
    v2, v4 = stage_stats(STAGES_V2), stage_stats(STAGES)
    W, H, x0, step = 900, 560, 90, 100
    fy_top, fy_bot, flo, fhi = 120, 170, 80, 100   # 최신성 칸
    ytop, ybot, lo, hi = 235, 435, 60, 100          # 답변 정확도 칸
    X = lambda i: x0 + i * step
    Y = lambda v: ybot - (v - lo) / (hi - lo) * (ybot - ytop)
    FY = lambda v: fy_bot - (v - flo) / (fhi - flo) * (fy_bot - fy_top)
    last = v4[6]
    b = [text(40, 34, f"정답지를 고친 뒤(v4)에도 답변 정확도는 {v4[0]['acc']:.0f}%에서 {last['acc']:.0f}%로 오르고, 최신성은 {v4[0]['fresh']:.0f}%에서 100%가 됐다", 18, INK, weight="bold")]
    # 범례
    b.append(f"<line x1='40' x2='64' y1='64' y2='64' stroke='{BLUE}' stroke-width='3'/><circle cx='52' cy='64' r='5' fill='{BLUE}'/>")
    b.append(text(72, 69, "답변 정확도 v4 (정답지 수정 후)", 13, MUTED))
    b.append(f"<line x1='290' x2='314' y1='64' y2='64' stroke='{V2_GREY}' stroke-width='2.5' stroke-dasharray='6 4'/><circle cx='302' cy='64' r='4' fill='{V2_GREY}'/>")
    b.append(text(322, 69, "v2 (수정 전)", 13, MUTED))
    b.append(f"<circle cx='428' cy='64' r='4' fill='white' stroke='{BLUE}' stroke-width='2'/>")
    b.append(text(440, 69, "같은 설정 2회의 각 실행(평균은 채운 점)", 13, MUTED))
    # 최신성 칸
    b.append(text(40, fy_top - 22, "최신성 (정답지 수정과 무관해 v2와 v4가 같다)", 13, INK, weight="bold"))
    for v in (80, 100):
        b.append(f"<line x1='60' x2='{X(7) + 30}' y1='{FY(v)}' y2='{FY(v)}' stroke='{GRID}'/>")
        b.append(text(54, FY(v) + 4, f"{v}%", 11, MUTED, "end"))
    pts = " ".join(f"{X(s['n'])},{FY(s['fresh'])}" for s in v4)
    b.append(f"<polyline points='{pts}' fill='none' stroke='{GREY}' stroke-width='2.5'/>")
    for s in v4:
        b.append(f"<circle cx='{X(s['n'])}' cy='{FY(s['fresh'])}' r='4' fill='{GREY}'><title>{s['n']} {escape(s['name'])} · 최신성 {s['fresh']:.1f}%</title></circle>")
    for n in (0, 3):
        b.append(text(X(n), FY(v4[n]["fresh"]) - 10, f"{v4[n]['fresh']:.1f}%", 12, MUTED, "middle"))
    # 답변 정확도 칸
    b.append(text(40, ytop - 28, "답변 정확도", 13, INK, weight="bold"))
    for v in range(lo, hi + 1, 10):
        b.append(f"<line x1='60' x2='{X(7) + 30}' y1='{Y(v)}' y2='{Y(v)}' stroke='{GRID if v > lo else '#9ca3af'}'/>")
        b.append(text(54, Y(v) + 4, f"{v}%", 11, MUTED, "end"))
    sep = (X(6) + X(7)) / 2
    b.append(f"<line x1='{sep}' x2='{sep}' y1='{fy_top - 30}' y2='{ybot + 50}' stroke='{MUTED}' stroke-dasharray='4 4'/>")
    b.append(text(X(7), fy_top - 22, "참고: k=10", 12, MUTED, "middle"))
    for series, color, w, dash, r in ((v2, V2_GREY, 2.5, " stroke-dasharray='6 4'", 4), (v4, BLUE, 3, "", 5)):
        for seg in (series[:7], series[7:]):  # top_k=10은 다른 설정이라 선을 잇지 않는다
            if len(seg) > 1:
                pts = " ".join(f"{X(s['n'])},{Y(s['acc'])}" for s in seg)
                b.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{w}'{dash}/>")
        for s in series:
            x = X(s["n"])
            ver = "v4" if color == BLUE else "v2"
            if s["runs"] > 1:
                for k, a in enumerate(s["accs"]):
                    b.append(f"<circle cx='{x}' cy='{Y(a)}' r='{r - 1}' fill='white' stroke='{color}' stroke-width='2'><title>{s['n']} {escape(s['name'])} · {'ab'[k]} 실행 {ver} {a:.1f}%</title></circle>")
            b.append(f"<circle cx='{x}' cy='{Y(s['acc'])}' r='{r}' fill='{color}'><title>{s['n']} {escape(s['name'])} · 답변 정확도 {ver} {s['acc']:.1f}%" + (f" (두 번 {min(s['accs']):.1f}~{max(s['accs']):.1f}%)" if s["runs"] > 1 else "") + "</title></circle>")
    for s4, s2 in zip(v4, v2):
        x = X(s4["n"])
        b.append(text(x, Y(max(s4["accs"])) - 12, f"{s4['acc']:.1f}%", 13, INK, "middle", "bold"))
        b.append(text(x, Y(min(s2["accs"])) + 22, f"{s2['acc']:.1f}%", 12, MUTED, "middle"))
        b.append(text(x, ybot + 24, str(s4["n"]) if s4["n"] < 7 else "k10", 14, INK, "middle", "bold"))
        b.append(text(x, ybot + 42, s4["name"], 12, MUTED, "middle"))
    b.append(text(40, H - 40, "전략 B. 정답지 수정(v4)은 A03·A04·A09·A14를 질문만 보고 고친 것이고, 15개 원본 답변을 기준선부터 모두 다시 채점했다. 세로축은 60%부터.", 12, MUTED))
    b.append(text(40, H - 22, "5·6단계와 k10은 같은 설정 2회 실행의 평균이다. 수정하지 않은 문항도 재채점으로 −4~+6%p 흔들려, 단계별 증감을 개선 효과로 단정하지 않는다.", 12, MUTED))
    b.append(text(40, H - 4, "회색 점선 아래 숫자는 v2(수정 전), 파란 선 위 숫자는 v4(수정 후). k10은 top_k=10 최종 설정이라 6단계와 선을 잇지 않았다.", 12, MUTED))
    return svg(W, H, "단계별 답변 정확도(v2 → v4)와 최신성", "".join(b),
               "전략 B의 0~6단계와 top_k=10 최종의 답변 정확도를 정답지 수정 전(v2)과 후(v4)로 비교하고, 최신성을 따로 보여 준다")


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
    for n, name, files in STAGES[:7]:  # top_k=10 실행은 히트맵에 넣지 않는다
        for k, f in enumerate(files):
            label = str(n) if len(files) == 1 else f"{n}-{'ab'[k]}"
            cols.append((label, SHORT[n] if k == 0 else "", f))
    data = [load(f) for _, _, f in cols]
    cw, rh, left, top = 62, 30, 250, 118
    n_rows = sum(len(g[1]) for g in groups)
    W, H = left + cw * len(cols) + 40, top + rh * n_rows + 90
    b = [text(40, 34, "v4 기준으로도 검색 문제로 틀리던 문항은 풀렸고, A16과 A05가 남았다", 18, INK, weight="bold")]
    b.append(text(40, 58, "O 정답 · △ 부분 · X 오답 (LLM 채점, 채점 기준 v4) · ※ 정답지를 고친 문항", 13, MUTED))
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
            b.append(text(left - 14, y + 20, i + ("※" if i in FIXED_IDS else ""), 14, INK, "end", "bold"))
            for j, rows in enumerate(data):
                v = verdict(rows[i])
                b.append(f"<rect x='{left + j * cw + 2}' y='{y + 2}' width='{cw - 4}' height='{rh - 4}' rx='4' fill='{CELL[v]}'>"
                         f"<title>{i} · {cols[j][0]}단계: {v}</title></rect>")
                b.append(text(left + j * cw + cw / 2, y + 21, SYM[v], 14, INK, "middle"))
            y += rh
    b.append(text(40, H - 36, "1~4단계는 1회 실행, 5·6단계는 같은 설정 2회(a, b). 1~4단계 열 아래 이름은 그 단계에서 추가한 것이다.", 12, MUTED))
    b.append(text(40, H - 18, "5·6단계의 a·b가 다른 문항은 노이즈로 본다. ※(A03·A04·A09·A14)는 정답지를 질문만 보고 고친 뒤 처음 버전부터 다시 채점한 값이다.", 12, MUTED))
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
    b.append(text(40, H - 10, "같은 설정으로 답변까지 평가하면(2회) 답변 정확도(v4) 72.0% → 82.0% / 76.0% (평균 79.0%). 파이프라인 전체에서는 차이가 없다.", 12, MUTED))
    return svg(W, H, "top_k별 검색 적중률", "".join(b), "top_k 3, 5, 10, 20에서 질문 그대로와 동의어 확장의 검색 적중률")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("stage_scores.svg", fig_stage_scores), ("item_heatmap.svg", fig_heatmap),
                     ("top_k_retrieval.svg", fig_topk)):
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("→", (OUT / name).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
