# 채용공고 기반 IT 직무 역량 분석 RAG 챗봇

점핏(Jumpit) 개발자 채용공고를 근거로 직무별 요구 역량을 답하고, 내 이력서와 비교해 부족한 역량을 알려주는 RAG 챗봇입니다.

## 진행 현황

| 단계 | 상태 | 문서 |
|---|---|---|
| 1. 주제 확정 | ✅ 완료 | [제안서](docs/proposal.pdf) |
| 2. 데이터 구축 | ✅ 완료 | [데이터 카드](data/README.md) |
| 3. 파싱·정제·청킹 | ✅ 완료 | [정제 규칙](docs/step3_cleaning_rules.md) · [청크 통계](docs/step3_chunk_stats.md) · [메타데이터](docs/step3_metadata.md) · [청크 수동 검사](docs/step3_chunk_review.md) |
| 4. 평가셋 완성 | ✅ 완료 | [평가셋 설명과 채점 기준](eval/README.md) |
| 5. 오답노트 | ✅ 완료 | [오답노트](docs/step5_error_notes.md) |
| 6. 개선 | ✅ 완료 | [실험 로그](docs/step6_experiment_log.md) · 그래프 `docs/figures/` |
| 7. 서비스화 | ✅ 완료 | [서비스화 기록](docs/step7_service.md) |
| 8. 최종 측정·리포트 | ✅ 완료 | [최종 리포트](docs/step8_final_report.md) · [최종 측정](docs/step8_final_measurement.md) |

## 결과 요약

같은 채점 기준(v4)으로 답변 정확도를 비교했습니다. 정답 1점, 부분 0.5점, 오답 0점의 평균입니다.

| 설정 | 답변 정확도 | 비고 |
|---|---|---|
| GPT 빈손 (공고 없이 답함) | 12.0% | 개발 문항 25개 |
| 고치기 전 (전략 B, top-5, 지시문 v1) | 72.0% | 개발 문항 25개 |
| **고친 후** | **94.0%·96.0%** | 개발 문항 25개, 같은 설정 2회 |
| 고친 후, 새 문항(홀드아웃) | 88.5% | 개발 때 보지 않은 문항 13개, 1회 |

- 홀드아웃은 코드를 동결(`freeze-final`)하고 문항을 잠근 뒤 한 번만 측정했고, 결과를 보고 고치지 않았습니다.
- 홀드아웃에서 틀린 2문항은 같은 회사에 공고가 여러 개일 때 검색이 정답 공고를 놓친 경우입니다. 둘 다 지어내지 않고 "확인되지 않는다"로 답했고, 고치지 않고 한계로 남겼습니다.
- 채점은 AI(gpt-6-luna)가 했고, 홀드아웃 15문항은 사람이 모두 확인해 판정이 일치했습니다 (15/15).
- 전체 과정은 [최종 리포트](docs/step8_final_report.md)에, 자세한 숫자와 읽는 법은 [최종 측정](docs/step8_final_measurement.md)에 있습니다.

## 문서

- [프로젝트 제안서](docs/proposal.pdf)
- [평가셋 설명](eval/README.md): 개발 문항 30개(`eval/eval_set.csv`), 홀드아웃 15개(`eval/holdout_set.csv`)
- 1단계 초안 평가셋(`eval/eval_set.xlsx`, 34문항)은 이력 확인용으로만 남겨 두었습니다.

## 문제 정의

- 공고마다 요구 기술·자격요건·우대사항이 달라 무엇부터 준비해야 할지 판단하기 어렵습니다.
- 여러 공고를 직접 찾아 읽고 비교하는 데 시간이 많이 듭니다.
- 일반 AI는 실제 공고가 아닌 일반론으로 답하거나, 마감일을 틀리고 공고에 없는 내용을 지어내기도 합니다.
  (ChatGPT 웹 검색 사용 시 5문항 중 4문항에서 마감일 오류·마감 공고 추천 확인 → 제안서 2-1)

## 타겟

취업·이직을 준비하는 신입~3년차 주니어 개발자

## 핵심 기능

| 기능 | 설명 | 단계 |
|---|---|---|
| 직무·경력별 공고 검색 | 직무·경력 조건으로 공고 필터링 | MVP · 구현 |
| 채용공고 기반 Q&A | 수집된 공고를 근거로 요구 역량·우대사항·전형 절차 질의응답 | MVP · 구현 |
| 특정 공고 상세 분석 | 선택한 공고 하나의 범위 안에서만 답변 | MVP · 구현 |
| 근거 기반 답변 | 답변마다 출처 공고·항목과 마감 여부 표시, 공고에 없는 내용은 "확인되지 않습니다" | MVP · 구현 |
| 이력서 기반 역량 분석 | 이력서와 공고 요구 역량을 비교해 부족한 역량 제시 | 확장 · 이번 범위에 넣지 않음 |

## 처리 흐름

1. 점핏 개발 직군 공고 수집 (신입~경력 3년 이하)
2. 데이터 정제 및 기술명 정규화 (예: 스프링 → Spring)
3. 항목 단위 Chunking(전략 B: 항목 단위, 800자 이상인 항목만 추가로 분할) + 공고 개요 청크 + 메타데이터 부여 (직무·경력·기업·공고ID·마감일·수집일·기술스택)
4. Embedding(`text-embedding-3-small`) → ChromaDB 저장
5. 질문 시 메타데이터 필터로 검색 범위 제한 (전체 / 직무·경력 필터 / 특정 공고)
6. 검색된 공고만 근거로 LLM 답변 생성
7. 인용한 근거 문장이 원문에 실제로 있는지 검증 후 답변 + 출처 + 마감 여부 제공

## 기술 스택

- Language: Python 3.13
- Vector DB: ChromaDB
- Embedding: OpenAI `text-embedding-3-small`
- LLM(답변·채점): OpenAI `gpt-6-luna`
- 화면: Streamlit · 질문 로그: SQLite

## 폴더 구조

```
├── docs/
│   ├── proposal.pdf             # 제안서
│   ├── step3_*.md               # 정제 규칙, 청크 통계, 메타데이터, 청크 수동 검사
│   ├── step5_error_notes.md     # 오답노트
│   ├── step6_experiment_log.md  # 실험 로그
│   ├── step7_service.md         # 서비스화 기록
│   ├── step8_final_measurement.md  # 최종 측정
│   ├── step8_final_report.md    # 최종 리포트 (전체 과정 정리)
│   └── figures/                 # 그래프
├── eval/
│   ├── README.md                # 평가셋 설명과 채점 기준
│   ├── eval_set.csv             # 개발 문항 30개
│   ├── holdout_set.csv          # 홀드아웃 15개 (최종 측정용, 잠금 후 수정 금지)
│   ├── eval_set.xlsx            # 1단계 초안 (이력용)
│   ├── results/                 # 평가 실행 결과 (summary.csv에 지표 한 줄씩)
│   └── usage_log.csv            # API 사용량과 비용 기록
├── data/
│   ├── README.md                # 데이터 카드 (출처, 문서 수, 제외 기준, 컬럼 설명)
│   ├── postings.csv             # 공고별 최신 상태 (서비스용)
│   ├── collection_summary.csv   # 수집 실행 기록
│   ├── eval_snapshot_2026-09-30/   # 평가용 고정 스냅샷 (수정 금지)
│   ├── posting_fields_*.csv     # 공고별 추출 칸 (재택·코딩테스트·기술 등)
│   ├── tech_synonyms.json       # 기술명 동의어 사전
│   ├── raw/                     # 공고 원문 (비공개)
│   ├── chunks/                  # 청크 (비공개)
│   └── service_log.db           # 질문 로그 (비공개)
├── src/
│   ├── collect_jumpit.py        # 점핏 공고 수집
│   ├── chunk_postings.py        # 청킹
│   ├── build_index.py           # 임베딩과 ChromaDB 색인
│   ├── extract_fields.py        # 공고별 칸 추출
│   ├── posting_status.py        # 기준일로 마감 여부 판정
│   ├── rag_engine.py            # 질문 1개를 처리하는 엔진
│   ├── app.py                   # Streamlit 화면
│   ├── log_db.py                # 질문 로그 DB
│   ├── run_eval.py              # 평가 자동 실행·채점
│   ├── check_engine.py          # 엔진 회귀 점검
│   ├── topk_retrieval.py        # top-k 비교 실험
│   └── make_figures.py          # 그래프 생성
├── .streamlit/config.toml       # 화면 테마
└── requirements.txt
```

## 평가

- 개발 문항 30개 ([eval/eval_set.csv](eval/eval_set.csv)): 정답형 20 · 문서에 없음 5 · 함정 5. 질문 + 정답 + 근거 위치(공고ID/항목) + 필수 요소로 구성
- 홀드아웃 15개 ([eval/holdout_set.csv](eval/holdout_set.csv)): 코드 동결 뒤 새로 쓰고 잠근 문항. 한 번만 측정
- 평가용 데이터는 9/30 스냅샷(사용 문서 101건)으로 고정하고, 평가 기준일은 2026-10-02로 고정해 마감 여부를 판정합니다.
- 지표 우선순위: 최신성 정확도 > 출처 정확도 > 답변 정확도 > 검색 정확도
- 자동 실행·채점은 `src/run_eval.py`가 하고, LLM 채점과 사람 채점의 일치율은 90% → 100%까지 맞췄습니다. 채점 기준은 [eval/README.md](eval/README.md)에 있습니다.
- 평가 실행에 쓴 API 비용은 `eval/usage_log.csv`에 기록합니다.

## 데이터 출처 및 이용

- 출처: [점핏](https://jumpit.saramin.co.kr) 개발자 채용공고
- 평가용 데이터: 2026-09-30 수집본 고정, 사용 문서 101건
- 서비스용 데이터: 2026-10-08 재수집, 사용 문서 140건 (진행 중 104 · 조기 마감 10 · 마감 26)
- 자세한 내용은 [데이터 카드](data/README.md)를 참고하세요.
- robots.txt상 공고 목록·상세 페이지는 수집 제한 경로가 아닙니다 (2026.09.29 확인).
- 사람인 회원약관 제19조에 따라 비상업 학습 목적으로만 사용하며, 공고 원문은 저장소에 공개하지 않습니다.

## 실행 방법

Windows, Python 3.13 기준입니다. 명령은 저장소 폴더에서 실행합니다.

### 1. 환경 준비

```bash
# 가상환경 만들기 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

프로젝트 폴더에 `.env` 파일을 만들고 OpenAI API 키를 넣습니다. `.env`는 `.gitignore`에 들어 있어 저장소에 올라가지 않습니다.

```
OPENAI_API_KEY=여기에_본인_키
```

### 2. 서비스용 데이터 만들기

공고 원문은 저장소에 없으므로 직접 수집해야 합니다. 순서대로 실행합니다.

```bash
python src/collect_jumpit.py            # 공고 수집 (다시 실행하면 새 공고 추가·변경 반영·마감 처리)
python src/chunk_postings.py service    # 청킹
python src/build_index.py service       # 임베딩과 ChromaDB 색인 (API 호출)
python src/extract_fields.py service    # 공고별 칸 추출 (API 호출)
python src/extract_fields.py service --career   # 본문 경력 수준 칸 추가 (API 호출)
```

수집은 점핏 서버에 요청을 보냅니다. 요청마다 간격을 두지만, 필요할 때만 실행하세요. 색인과 칸 추출에는 140건 기준 약 $0.04의 API 비용이 들었습니다 (`docs/step7_service.md`).

### 3. 챗봇 실행

```bash
python -m streamlit run src/app.py
```

사이드바에서 직무·경력 필터와 "마감 포함" 스위치를 고르거나 공고 하나를 선택해 질문합니다. 답변에는 근거 공고 카드, 마감 여부, 점핏 링크가 함께 표시되고, 질문은 `data/service_log.db`에 기록됩니다.

### 평가 다시 돌리기

평가는 9/30 수집본의 공고 원문과 청크(비공개)가 있어야 같은 조건으로 재현됩니다. 공고는 마감되면 내려가므로 지금은 같은 원문을 다시 받기 어렵습니다. 최종 설정 명령과 결과 파일은 [최종 측정](docs/step8_final_measurement.md)과 `eval/results/`에 남겨 두었습니다.
