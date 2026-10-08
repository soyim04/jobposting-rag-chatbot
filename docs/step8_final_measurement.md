# 8단계 최종 측정 (코드 동결 · 홀드아웃)

작성일 2026-10-08. 최종 리포트에 쓸 숫자를 정하는 측정이다. 규칙은 코드를 멈추고, 문항을 잠그고, 측정만 한다.

## 1. 코드 동결

| 항목 | 내용 |
|---|---|
| 동결 시점 | 2026-10-08 |
| **동결 커밋** | `a990a4b` (전체 `a990a4bb90b99936204bc306612a38f5482db245`), 태그 `freeze-final` |
| 규칙 | 이 커밋 이후 **검색·프롬프트·규칙을 고치지 않는다**. 홀드아웃에서 틀린 문항도 고치지 않고 실패 사례로 쓴다 |
| 마지막 변경 | 평가 도구에 `--eval-set` 옵션 추가(홀드아웃 문항을 같은 방식으로 돌리기 위함). 기본값은 기존과 같고 검색·프롬프트·규칙은 바꾸지 않았다 |
| 채점 기준 | **v4로 동결**. 더 이상 바꾸지 않는다 |
| 평가 기준일 | 2026-10-02 (고정) |
| 모델 | 답변·채점 `gpt-6-luna`, 임베딩 `text-embedding-3-small` |
| 실행 환경 | Python 3.13.15, chromadb 1.5.9, openai 3.23.0 |

**동결 대상**: `src/run_eval.py`, `src/rag_engine.py`, `src/build_index.py`, `src/chunk_postings.py`, `src/extract_fields.py`, `src/posting_status.py`, `data/tech_synonyms.json`, `data/posting_fields_eval.csv`, `eval/eval_set.csv`(개발 문항 30개와 정답지).
**동결 뒤에도 바뀔 수 있는 것**(파이프라인과 무관): `src/app.py`, `src/make_figures.py`, 문서, 새 홀드아웃 문항 파일, 측정 결과 파일.

### 비공개 데이터의 동결 확인값

색인과 청크는 저장소에 올라가지 않아서 해시와 개수로 기록한다. 측정 직전과 직후에 같은 값인지 확인한다.

| 대상 | 값 |
|---|---|
| `data/chunks/eval/strategy_b.jsonl` | sha256 앞 16자 `8d816fe260fa89d9` |
| 컬렉션 `eval-b-text-embedding-3-small` | 청크 729개 |
| `data/eval_snapshot_2026-09-30/postings.csv` | `0da470e6a7eea0d8` |
| `data/posting_fields_eval.csv` | `dfbc459e2c5e7b1b` |
| `data/tech_synonyms.json` | `a38e9d859d481ec8` |
| `eval/eval_set.csv` | `d9e6d9e29831af12` |

### 동결을 지켰는지 확인하는 방법

측정이 끝난 뒤 아래 명령의 출력이 비어 있어야 한다.

```
git diff freeze-final..HEAD -- src/run_eval.py src/rag_engine.py src/build_index.py src/chunk_postings.py src/extract_fields.py src/posting_status.py data/tech_synonyms.json data/posting_fields_eval.csv eval/eval_set.csv
```

## 2. 최종 설정 (고친 후)

6단계까지 실험으로 정한 설정이다. 평가 도구로 아래 옵션을 켜서 개발 문항과 홀드아웃을 같은 방식으로 돌린다.

```
.venv\Scripts\python.exe src\run_eval.py --strategy B --top-k 10 --full-posting --synonyms --structured --diff-notes --diff-relevant-only --prompt v2 --per-posting 2 [--eval-set eval\holdout_set.csv] --label a
```

## 3. 측정 계획

| 순서 | 내용 | 상태 |
|---|---|---|
| 1 | 도구 정비(`--eval-set`) | 완료 (`a990a4b`) |
| 2 | 코드 동결 | 완료 (이 문서) |
| 3 | 홀드아웃 문항 작성(15개, 시스템에 돌리기 전에 정답지까지 확정해 커밋) | 예정 |
| 4 | 측정: GPT 빈손 v4 재채점, 고친 후(개발 문항) 2회, 고친 후(홀드아웃) **딱 1회** | 예정 |
| 5 | 숫자 표와 실패 사례 정리 | 예정 |

고치기 전은 기존 기준선 B(v4 재채점 결과)를 그대로 쓴다.

## 4. 알려진 한계 (미리 적어 둔다)

- 홀드아웃 문항은 새것이지만 공고 모음은 개발 때와 같은 9/30 스냅샷이다. 기술 동의어 사전과 조건 해석 규칙은 개발 문항의 실패를 보고 만들었다.
- 채점 LLM의 판정은 같은 답변이라도 한두 문항씩 흔들린다(마감일·상태 덧붙임을 감점할지 판정이 갈린 사례가 확인됨). 채점 기준은 더 바꾸지 않기로 했으므로 한계로만 기록한다.
