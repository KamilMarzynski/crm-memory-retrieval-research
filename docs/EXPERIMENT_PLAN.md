# Experiment Plan: Improving Memory Retrieval

## Priority 1: Fix Data Pipeline (Blocking)

### Step 1.1: Regenerate Test Cases
```bash
# Current state: Test cases reference phase0 memory IDs
# Target state: Test cases reference phase1 memory IDs

uv run python scripts/phase1/test_cases.py
```

### Step 1.2: Verify Alignment
```bash
# Check that ground truth IDs now exist in database
uv run python -c "
import json
import sqlite3
from pathlib import Path

db = sqlite3.connect('data/phase1/memories/memories.db')
db_ids = set(row[0] for row in db.execute('SELECT id FROM memories'))

for tc_file in Path('data/phase1/test_cases').glob('*.json'):
    with open(tc_file) as f:
        tc = json.load(f)
    gt_ids = set(tc.get('ground_truth_memory_ids', []))
    missing = gt_ids - db_ids
    if missing:
        print(f'❌ {tc_file.stem}: {len(missing)} missing IDs')
    else:
        print(f'✅ {tc_file.stem}: {len(gt_ids)} memories OK')
"
```

### Step 1.3: Run Baseline Experiment
```bash
uv run python scripts/phase1/experiment.py --all
```

Expected: Recall should now be > 0% (we'll see actual retrieval performance)

---

## Priority 2: Query Prompt Improvement

### Experiment 2A: Pattern Vocabulary Query Prompt

**Hypothesis:** Replacing domain terms with pattern terms will improve recall.

**Implementation:**
1. Create `scripts/phase1/experiment_v2.py` with new query prompt
2. Run experiments with both prompts
3. Compare recall

**New prompt key changes:**
- Query length: 10-25 words (vs current 2-3 sentences)
- No domain terms (authentication → decorator, pagination → numeric parameters)
- Structured format: [structure] + [pattern] + [gap]

**Test:**
```bash
# Run with current prompt
uv run python scripts/phase1/experiment.py --all
# Save results as baseline

# Run with new prompt (modify experiment.py)
uv run python scripts/phase1/experiment.py --all
# Compare recall
```

### Experiment 2B: Query Count and Diversity

**Hypothesis:** More diverse queries improve recall through coverage.

**Variations to test:**
- 5 queries (current-ish)
- 10 queries
- 15 queries
- 20 queries

**Implementation:** Modify `MAX_QUERIES_PER_EXPERIMENT` and prompt

---

## Priority 3: Multi-Query Fusion

### Experiment 3A: Reciprocal Rank Fusion

**Hypothesis:** Combining results across queries improves recall.

**Implementation:**
```python
# Add to db.py or experiment.py
def search_with_rrf(db_path: str, queries: List[str], k: int = 5, rrf_k: int = 60):
    """
    Search with multiple queries and combine using Reciprocal Rank Fusion.

    RRF score for document d across queries Q:
    score(d) = sum over q in Q of: 1 / (rrf_k + rank(d, q))
    """
    from collections import defaultdict

    scores = defaultdict(float)
    seen_memories = {}

    for query in queries:
        results = search_memories(db_path, query, limit=k * 2)
        for rank, result in enumerate(results):
            mem_id = result["id"]
            scores[mem_id] += 1.0 / (rrf_k + rank + 1)
            if mem_id not in seen_memories:
                seen_memories[mem_id] = result

    # Sort by combined score
    top_ids = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [seen_memories[mid] for mid, _ in top_ids if mid in seen_memories]
```

**Test:**
```bash
# Modify experiment.py to use search_with_rrf
# Compare recall vs individual query approach
```

---

## Priority 4: Hybrid Search (BM25 + Vector)

### Experiment 4A: Add FTS5 to Phase 1 Database

**Implementation:**
```python
# Modify db.py to add FTS5 index alongside vec0

def create_database(db_path: str) -> None:
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()

        # Existing tables...

        # Add FTS5 for keyword search
        cur.execute("""
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                id,
                situation_description,
                lesson,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Populate FTS
        cur.execute("""
            INSERT INTO memories_fts(id, situation_description, lesson)
            SELECT id, situation_description, lesson FROM memories
        """)
```

### Experiment 4B: Hybrid Scoring

**Implementation:**
```python
def hybrid_search(db_path: str, query: str, alpha: float = 0.7, limit: int = 10):
    """
    Combine vector similarity and BM25 keyword search.

    alpha: Weight for vector score (0=BM25 only, 1=vector only)
    """
    # Vector search
    vec_results = search_memories_vector(db_path, query, limit=limit*2)
    vec_scores = {r["id"]: 1 - r["distance"] for r in vec_results}  # Convert distance to similarity

    # BM25 search
    bm25_results = search_memories_fts(db_path, query, limit=limit*2)
    bm25_max = max(r["bm25_score"] for r in bm25_results) if bm25_results else 1
    bm25_scores = {r["id"]: r["bm25_score"] / bm25_max for r in bm25_results}

    # Combine
    all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for mid in all_ids:
        vs = vec_scores.get(mid, 0)
        bs = bm25_scores.get(mid, 0)
        combined[mid] = alpha * vs + (1 - alpha) * bs

    # Return top results
    top_ids = sorted(combined.items(), key=lambda x: -x[1])[:limit]
    return [get_memory_by_id(db_path, mid) for mid, _ in top_ids]
```

**Grid search for alpha:**
```python
alphas = [0.3, 0.5, 0.7, 0.9]
for alpha in alphas:
    recall = run_experiments_with_hybrid(alpha)
    print(f"alpha={alpha}: recall={recall}")
```

---

## Priority 5: Memory Extraction Improvements

### Experiment 5A: Add Searchable Tags

**Implementation in `build_memories.py`:**
```python
# After extracting situation and lesson, extract tags
tags = call_openrouter(
    api_key=api_key,
    model=model,
    messages=_prompt_searchable_tags(situation, code_snippet),
    temperature=0.0,
    max_tokens=100,
)
tags_list = json.loads(tags) if tags else []

memory = {
    "id": _stable_id(...),
    "situation_description": situation,
    "searchable_tags": tags_list,  # NEW
    "lesson": lesson,
    "metadata": {...}
}
```

**Test:**
1. Rebuild memories with tags
2. Rebuild database with tags in embedding
3. Run experiments
4. Compare recall

---

## Priority 6: Alternative Embedding Models

### Experiment 6A: Compare Embedding Models

**Models to test:**
1. `mxbai-embed-large` (current) - 1024 dim
2. `nomic-embed-text` - 768 dim
3. `all-minilm-l6-v2` - 384 dim (fast baseline)
4. `voyage-code-2` (if API available) - code-specific

**Implementation:**
```python
# Create db.py variant for each model
# Rebuild database with different embeddings
# Run same experiments
# Compare recall
```

---

## Metrics Dashboard

For each experiment, track:

| Metric | Formula | Target |
|--------|---------|--------|
| Recall@5 | (retrieved ∩ ground_truth) / ground_truth | > 60% |
| Recall@10 | Same, with k=10 | > 80% |
| MRR | Mean of 1/rank(first ground truth) | > 0.4 |
| Precision@5 | (retrieved ∩ ground_truth) / retrieved | > 30% |
| False Positive Rate | (retrieved - ground_truth) / retrieved | < 70% |

---

## Experiment Tracking Template

```json
{
  "experiment_id": "exp_YYYYMMDD_HHMMSS",
  "hypothesis": "...",
  "changes": ["..."],
  "baseline_recall": 0.0,
  "experiment_recall": 0.0,
  "improvement": 0.0,
  "notes": "...",
  "next_steps": ["..."]
}
```

---

## Timeline

### Day 1 (Critical)
- [ ] Fix data pipeline (Priority 1)
- [ ] Run baseline experiments
- [ ] Document baseline metrics

### Day 2-3 (High Impact)
- [ ] Implement new query prompt (Priority 2)
- [ ] A/B test prompt changes
- [ ] Implement RRF fusion (Priority 3)

### Day 4-5 (Medium Impact)
- [ ] Add hybrid search (Priority 4)
- [ ] Grid search for alpha parameter
- [ ] Add searchable tags (Priority 5)

### Week 2 (Research)
- [ ] Test alternative embeddings (Priority 6)
- [ ] Fine-tuning experiments if needed
- [ ] Final evaluation and documentation

---

## Success Criteria

| Milestone | Recall Target | Status |
|-----------|---------------|--------|
| Baseline (after data fix) | > 10% | ⏳ |
| After prompt improvement | > 40% | ⏳ |
| After RRF fusion | > 50% | ⏳ |
| After hybrid search | > 60% | ⏳ |
| Final system | > 70% | ⏳ |
