PY ?= .venv/bin/python

.PHONY: setup data index index-bm25 index-local probe test floor baseline retrieval eval eval-quick safety api ui up gold

setup:            ## create venv and install
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e ".[dev]"

data:             ## download 10-Ks + XBRL from EDGAR, split into Items
	$(PY) -m finsight.data.ingest

gold:             ## (re)build eval/gold.jsonl - only before the Week-1 freeze!
	$(PY) -m eval.build_gold

index:            ## hybrid index (needs embeddings endpoint)
	$(PY) -m finsight.retrieval.index --strategy table

index-bm25:       ## BM25-only index (no endpoint needed)
	$(PY) -m finsight.retrieval.index --strategy table --no-dense

index-local:      ## dense index with a local CPU model (no quota; needs `pip install .[rerank]`)
	EMBED_BACKEND=local EMBED_MODEL=BAAI/bge-small-en-v1.5 $(PY) -m finsight.retrieval.index --strategy table

probe:            ## which JSON modes does the course endpoint support?
	$(PY) -m finsight.llm --probe

test:
	$(PY) -m pytest -q

floor:            ## what rules + XBRL + ratio tool answer with NO model (no endpoint needed)
	EMBED_BACKEND=none $(PY) -m eval.deterministic_eval

baseline:         ## Week 1: 3 prompting strategies x closed/oracle context
	$(PY) -m eval.baselines --modes closed oracle naive_rag --limit 40

retrieval:        ## Week 2: recall@k ablation
	$(PY) -m eval.retrieval_eval

eval-quick:       ## daily 30-question subset (quota friendly)
	$(PY) -m eval.run_eval --limit 30 --seed 1

eval:             ## full 100-question eval with LLM judge
	$(PY) -m eval.run_eval

safety:           ## Week 4: injection + PII tests
	$(PY) -m eval.safety_eval

api:
	$(PY) -m uvicorn finsight.api:app --reload --port 8000

ui:
	.venv/bin/streamlit run ui/app.py

up:
	docker compose up --build
