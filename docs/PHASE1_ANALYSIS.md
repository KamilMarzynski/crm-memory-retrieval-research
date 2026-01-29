# Phase 1 Retrieval Analysis & Improvement Plan

## Executive Summary

**Current recall: 0%** across all test cases (15 memories missed out of 15).

This document analyzes the root causes and proposes concrete improvements for memory retrieval.

---

## Critical Finding: Data Pipeline Mismatch

### The Problem

The ground truth memory IDs in test cases **do not exist in the database**.

| Test Case | Ground Truth IDs | Exist in DB? |
|-----------|------------------|--------------|
| VAPI-766 | `mem_223a2c9c63ca`, `mem_33f17f272499`, `mem_d1f5c76dcd0f` | ❌ No |
| VAPI-561 | `mem_2fa179b33eed`, `mem_66cd4c4e67cd` | ❌ No |
| VAPI-652 | `mem_d25d2ebb8a46`, `mem_d6c9096ae998` | ❌ No |
| VAPI-724 | `mem_03b1f00dd4f8`, `mem_0b3956816ca5`, `mem_35ae9bbd59ad` | ❌ No |
| VAPI-732 | `mem_25d675632b5c`, `mem_340a222d7430`, `mem_e7b38efb1384`, `mem_ea093e7da387`, `mem_f3edd91ce177` | ❌ No |

**Memories in database:** `mem_06c1b014634b`, `mem_1f53e962c7d6`, `mem_253eb74f43d0`, etc. (completely different set)

### Root Cause

Memory IDs are generated from content hash (`_stable_id` function):
```python
def _stable_id(raw_comment_id: str, situation: str, lesson: str) -> str:
    h = hashlib.sha1((raw_comment_id + "\n" + situation + "\n" + lesson).encode("utf-8")).hexdigest()
    return f"mem_{h[:12]}"
```

Since Phase 0 used `situation_variants` (3 paraphrases) and Phase 1 uses `situation_description` (single description), the content differs → different hashes → different IDs.

### Immediate Fix Required

**Run the test case generator after building memories:**
```bash
uv run python scripts/phase1/test_cases.py
```

This will regenerate test cases with correct memory IDs matching the Phase 1 database.

---

## Deeper Analysis: Even With Correct Data, Semantic Gap Exists

After fixing the data mismatch, we still face a fundamental semantic gap between queries and memories.

### Query vs Memory Examples

**Generated Query (VAPI-732):**
> "Authentication middleware replaced with passport strategy pattern. Multiple strategies registered but session configuration set per-decorator instead of globally in server settings."

**Actual Memory (mem_b7ab7df9b170):**
> "Decorator applying configuration option at multiple levels (both decorator and server-level settings). Redundant configuration creates maintenance burden when option values need to change or conflict."

**Why they don't match:**
1. Query mentions specific domain terms: "passport", "authentication", "middleware"
2. Memory uses generic pattern language: "decorator", "configuration option", "multiple levels"
3. Both describe the **same issue** but with completely different vocabulary

### Vocabulary Distribution Analysis

**Queries tend to use:**
- Domain-specific terms: "pagination", "authentication", "enum type", "API parameter"
- Implementation details: "passport strategy", "query parameter model", "service class"
- Action-oriented language: "refactored from", "replaced with", "introduced"

**Memories tend to use:**
- Pattern-oriented terms: "optional chaining", "nullable", "conditional logic"
- Structure-oriented: "decorator", "method", "mapper", "test suite"
- Gap-oriented: "missing", "lacks", "redundant", "inconsistent"

---

## Root Cause Analysis

### 1. Asymmetric Encoding Problem

The core issue is that memories and queries are encoded from fundamentally different perspectives:

```
Code Review Comment → Memory (What pattern does this represent?)
                    ↓
              [SEMANTIC GAP]
                    ↑
PR Context + Diff → Query (What might be relevant here?)
```

The memory extraction focuses on **reusable patterns**, while query generation focuses on **what's in the code**.

### 2. Abstraction Level Mismatch

| Layer | Memory | Query |
|-------|--------|-------|
| High-level | "Decorator applying configuration..." | "Authentication middleware..." |
| Mid-level | "...at multiple levels" | "...passport strategy pattern" |
| Low-level | N/A (abstracted away) | "session configuration set per-decorator" |

Memories abstract away domain details; queries preserve them.

### 3. Query Generation Prompt Issues

Current prompt asks to generate "situation-like" queries but:
- Allows 2-3 sentences (too long for vector similarity)
- Encourages domain terms ("include domain terms if prominent")
- No explicit guidance on pattern vocabulary

### 4. Single-Point Embedding

Both queries and memories are embedded as single vectors. This loses nuance:
- Long descriptions get "averaged out"
- Key technical terms may be diluted
- No opportunity for partial/fuzzy matching

---

## Proposed Improvements

### Phase A: Quick Wins (No Model Changes)

#### A1. Fix Data Pipeline
```bash
# Rebuild test cases with current memory IDs
uv run python scripts/phase1/test_cases.py
# Verify alignment
python -c "
import json
from pathlib import Path
for tc in Path('data/phase1/test_cases').glob('*.json'):
    data = json.load(open(tc))
    print(f'{tc.stem}: {len(data.get(\"ground_truth_memory_ids\", []))} memories')
"
```

#### A2. Improve Query Generation Prompt

**Current prompt issues:**
- "2-3 short sentences" → too verbose for embedding
- "include domain terms if prominent" → creates vocabulary mismatch

**Proposed prompt revision:**
```
TASK: Generate retrieval queries to find relevant code review patterns.

CRITICAL: Your queries must use PATTERN vocabulary, not DOMAIN vocabulary.
The database stores situations like:
- "Test suite for mapper method with conditional logic on optional nested object properties."
- "Decorator applying configuration option at multiple levels."
- "String split operation on potentially empty value without validation."

YOUR QUERIES should describe the same patterns you see in the diff:
- If you see passport authentication → query: "decorator configuration at multiple levels"
- If you see enum values → query: "constant array values inline instead of class-level"
- If you see pagination → query: "method accepting numeric parameters without validation"

RULES:
- Each query: 10-20 words (one sentence)
- Start with structure: "Test suite...", "Mapper method...", "Service class..."
- Include pattern: "optional chaining", "conditional logic", "null check"
- Include gap: "missing", "lacks", "inconsistent", "redundant"
- NO domain terms: NO "pagination", "authentication", "user", "payment"
- Generate 8-12 queries covering different patterns seen in the diff
```

#### A3. Multi-Query Fusion

Instead of retrieving with each query independently, combine top results:
```python
def search_with_fusion(queries: List[str], db_path: str, k: int = 5) -> List[Dict]:
    """Reciprocal Rank Fusion across multiple queries."""
    all_scores = {}

    for query in queries:
        results = search_memories(db_path, query, limit=k*2)
        for rank, result in enumerate(results):
            mem_id = result["id"]
            # RRF formula: 1 / (k + rank)
            score = 1.0 / (60 + rank)  # k=60 is standard
            all_scores[mem_id] = all_scores.get(mem_id, 0) + score

    # Sort by combined score and return top k
    sorted_ids = sorted(all_scores.items(), key=lambda x: -x[1])[:k]
    return [get_memory_by_id(db_path, mid) for mid, _ in sorted_ids]
```

### Phase B: Memory Structure Improvements

#### B1. Add Searchable Tags During Extraction

Modify `build_memories.py` to generate searchable tags:
```python
def _prompt_tags(situation: str, code_snippet: str) -> List[Dict[str, str]]:
    """Extract 3-5 searchable tags from the situation."""
    system = """Extract 3-5 short tags (2-4 words each) that capture:
    1. Code structure: "test file", "mapper method", "decorator"
    2. Pattern type: "optional chaining", "null check", "conditional logic"
    3. Gap type: "missing test", "redundant check", "inconsistent handling"

    Tags should be searchable terms that someone reviewing similar code would use.
    Output ONLY a JSON array of strings."""
    # ...
```

Memory schema becomes:
```json
{
  "id": "mem_xxx",
  "situation_description": "...",
  "searchable_tags": ["mapper method", "conditional logic", "missing test case"],
  "lesson": "...",
  "metadata": {...}
}
```

#### B2. Multi-Field Embedding

Embed memories with concatenated fields:
```python
def get_memory_embedding(memory: Dict) -> List[float]:
    """Create searchable embedding from multiple fields."""
    text_parts = [
        memory.get("situation_description", ""),
        " | ".join(memory.get("searchable_tags", [])),
        memory.get("lesson", ""),
    ]
    combined = " ".join(text_parts)
    return get_embedding(combined)
```

### Phase C: Hybrid Retrieval

#### C1. Add BM25/Keyword Search

Combine vector similarity with keyword matching:
```python
def hybrid_search(db_path: str, query: str, alpha: float = 0.7) -> List[Dict]:
    """
    Hybrid search combining vector similarity and BM25.

    Args:
        alpha: Weight for vector score (1-alpha for BM25)
    """
    vector_results = search_memories_vector(db_path, query, limit=20)
    bm25_results = search_memories_fts(db_path, query, limit=20)

    # Normalize and combine scores
    combined = {}
    for r in vector_results:
        combined[r["id"]] = alpha * (1 - r["distance"])  # Convert distance to similarity
    for r in bm25_results:
        combined[r["id"]] = combined.get(r["id"], 0) + (1 - alpha) * r["bm25_score"]

    return sorted(combined.items(), key=lambda x: -x[1])[:10]
```

#### C2. Query Expansion with HyDE

Generate hypothetical memory descriptions from queries:
```python
def expand_query_with_hyde(query: str) -> str:
    """Generate a hypothetical memory that would answer this query."""
    prompt = f"""
    Given this code review query:
    {query}

    Write a hypothetical "situation description" that a code review memory database would contain.
    Use technical pattern language, not domain language.
    Format: 2 sentences, ~50 words, describing WHEN and WHAT.
    """
    return call_llm(prompt)
```

### Phase D: Better Embeddings

#### D1. Consider Code-Specific Embedding Models

Options to evaluate:
1. **CodeBERT** - Microsoft's code understanding model
2. **UniXcoder** - Unified cross-modal pre-training
3. **Voyage-code-2** - Voyage AI's code embedding
4. **Cohere embed-v3** - Multi-lingual with code support

#### D2. Fine-Tuning Strategy

If needed, fine-tune embedding model on:
- Positive pairs: (memory_situation, original_code_snippet)
- Negative pairs: (memory_situation, random_code_snippet)
- Hard negatives: (memory_situation, similar_but_wrong_pattern)

---

## Evaluation Framework

### Metrics to Track

1. **Recall@K**: % of ground truth memories in top K results
2. **MRR (Mean Reciprocal Rank)**: Average 1/rank of first ground truth
3. **Precision@K**: % of top K results that are ground truth
4. **False Positive Rate**: % of retrieved memories with distance > threshold

### Ablation Experiments

| Experiment | Description | Expected Impact |
|------------|-------------|-----------------|
| A1 | Fix data pipeline | Enable measurement |
| A2 | Improved query prompt | +20-40% recall |
| A3 | Multi-query fusion | +10-20% recall |
| B1 | Searchable tags | +15-25% recall |
| C1 | Hybrid retrieval | +10-15% recall |
| C2 | HyDE expansion | +10-20% recall |

---

## Implementation Priority

### Immediate (Today)
1. ✅ Fix data pipeline (run `test_cases.py`)
2. Rerun experiments to establish baseline

### Short-term (This Week)
3. Revise query generation prompt (A2)
4. Implement multi-query fusion (A3)
5. Measure improvement

### Medium-term (Next Week)
6. Add searchable tags to memory extraction (B1)
7. Implement hybrid search (C1)
8. Evaluate code-specific embeddings (D1)

### Long-term (Research)
9. HyDE query expansion (C2)
10. Fine-tuning if needed (D2)

---

## Appendix: Example Query-Memory Pairs

### Should Match (Same Pattern, Different Words)

**Query:** "Authentication middleware replaced with passport strategy pattern. Multiple strategies registered but session configuration set per-decorator instead of globally in server settings."

**Memory:** "Decorator applying configuration option at multiple levels (both decorator and server-level settings). Redundant configuration creates maintenance burden when option values need to change or conflict."

**Pattern:** Configuration duplication across levels

---

**Query:** "Mapper derives a boolean flag from an optional nested property without null-checking the parent object."

**Memory:** "Test suite for mapper method with conditional logic on optional nested object properties. Missing test case for completely undefined parent object."

**Pattern:** Optional nested property handling

---

### Currently Retrieved (Wrong Pattern)

**Query:** "Authentication middleware replaced with passport strategy pattern..."

**Retrieved:** "Decorator parameter description mismatched with actual parameter purpose."

**Why wrong:** Both mention "decorator" but patterns are completely different (configuration duplication vs documentation mismatch)
