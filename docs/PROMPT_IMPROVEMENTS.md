# Prompt Engineering Improvements for Memory Retrieval

This document contains specific prompt improvements for both memory extraction and query generation, incorporating feedback from prompt engineering review.

**Version:** 3.0
**Last Updated:** 2026-01-29

---

## Key Design Principles

Based on expert review, these principles guide all prompt improvements:

1. **Length Alignment**: Memory and query outputs must be similar length for vector similarity to work (25-60 words for both)
2. **Example-Driven**: Include real memory examples in query prompts to ground the LLM in actual vocabulary
3. **Flexible Structure**: Provide key elements but allow natural ordering (not rigid templates)
4. **Hybrid Vocabulary**: Allow domain terms when they add semantic clarity
5. **Measurable**: Track both precision and recall; use embedding distance for confidence

---

## 1. Memory Extraction Improvements

### Current Issues with `build_memories.py`

The current situation extraction prompt produces descriptions that are:
1. Too abstract for retrieval
2. Missing searchable structural terms
3. Inconsistent in format/style
4. Wrong length (40-120 words vs queries at 10-25 words) - **critical mismatch**

### Proposed Memory Extraction Prompt (v3)

```python
# Prompt version for tracking
MEMORY_PROMPT_VERSION = "situation_v3.0"

def _prompt_situation_v3(context: str, c: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract searchable situation description from code review comment.

    Version: 3.0
    Output length: 25-60 words (aligned with query length)
    """
    code = (c.get("code_snippet") or "").strip()
    user_note = (c.get("user_note") or "").strip()
    file = c.get("file", "")
    severity = c.get("severity", "info")
    comment = c.get("message", "")

    system = """You extract reusable code review patterns for a vector search database.

Your output will be retrieved by semantic similarity, so focus on:
1. Technical patterns (optional chaining, null checks, conditional logic)
2. Code structure context (test file, mapper, service, decorator)
3. The gap or issue (missing, lacks, inconsistent, redundant)

Write 1-2 sentences (25-60 words) that naturally incorporate these elements.
The elements can appear in any order that reads naturally.

Do NOT include:
- Solutions, advice, or "should" statements
- File names, variable names, or code identifiers
- Generic statements without specific patterns

GOOD EXAMPLES:
"Test file for mapper method accepting optional object. Missing test case for completely undefined parent object; only tests undefined nested properties."
"Service method using optional chaining on nested properties. Early return may skip downstream validation when parent object is null."
"Decorator applying configuration at multiple levels. Session setting duplicated between decorator parameter and server-level config."

BAD EXAMPLES:
"The UserMapper.ts file has a bug." (includes file name, too vague)
"Add null check before accessing user.profile." (includes solution)
"Be careful with optional properties." (too vague, no pattern)
"Authentication middleware needs better error handling." (domain term without pattern context)"""

    user = f"""FILE: {file}
SEVERITY: {severity}

CODE:
{code[:800] if code else "(none)"}

REVIEW COMMENT:
{comment}

{f"ADDITIONAL CONTEXT: {user_note}" if user_note else ""}

Extract the reusable pattern (25-60 words). Output ONLY the description text."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

### Key Changes from v2:
- **Removed hardcoded structure detection** - LLM can infer from file path
- **Aligned length to 25-60 words** (was 40-120) to match query length
- **Relaxed rigid two-sentence structure** - elements can appear naturally
- **Added prompt versioning** for tracking

### Add Searchable Tags Extraction

```python
TAGS_PROMPT_VERSION = "tags_v3.0"

def _prompt_searchable_tags(situation: str, code_snippet: str) -> List[Dict[str, str]]:
    """
    Generate searchable tags from a memory situation.

    Version: 3.0
    Output: 3-5 tags, 2-4 words each
    """
    system = """Extract short searchable tags from code review patterns.

Each tag should be 2-4 words that someone might search for.

TAG CATEGORIES:
1. Structure tags: "test file", "mapper method", "service class", "decorator", "repository"
2. Pattern tags: "optional chaining", "null check", "conditional logic", "nested object", "enum handling"
3. Gap tags: "missing test", "inconsistent handling", "redundant check", "copy-paste error", "no validation"

OUTPUT: JSON array of 3-5 strings.

EXAMPLES:
["test file", "optional nested object", "missing edge case"]
["mapper method", "conditional logic", "boolean flag", "untested branch"]
["decorator", "configuration levels", "redundant setting"]"""

    user = f"""SITUATION:
{situation}

CODE CONTEXT:
{code_snippet[:500] if code_snippet else "(none)"}

Extract 3-5 searchable tags. Output ONLY a JSON array."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

---

## 2. Query Generation Improvements

### Current Issues with `experiment.py` Query Prompt

1. **Length mismatch**: Generates 2-3 sentence queries (50-80 words) vs memories at different length
2. Encourages domain-specific terms without guidance on when they're appropriate
3. No real examples from the actual database
4. No self-verification step

### Proposed Query Generation Prompt (v3)

**Critical improvement**: Include actual memory examples to ground the LLM in your database's vocabulary.

```python
QUERY_PROMPT_VERSION = "query_v3.0"

def _build_query_generation_prompt_v3(
    context: str,
    filtered_diff: str,
    sample_memories: List[Dict[str, Any]]  # NEW: Pass actual memories
) -> List[Dict[str, str]]:
    """
    Build improved prompt for pattern-based query generation.

    Version: 3.0
    Output length: 20-50 words per query (aligned with memory length)
    Requires: sample_memories from database for grounding
    """
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n... (truncated)"
    if len(filtered_diff) > MAX_DIFF_LENGTH:
        filtered_diff = filtered_diff[:MAX_DIFF_LENGTH] + "\n... (truncated)"

    # Format actual memory examples (critical for semantic alignment)
    memory_examples = "\n".join([
        f"- \"{m['situation_description']}\""
        for m in sample_memories[:5]
    ])

    system = f"""You generate search queries for a code review memory database.

The database contains patterns like these REAL EXAMPLES:
{memory_examples}

Your queries MUST sound like these entries to achieve semantic match.

QUERY RULES:
- Each query: 20-50 words (1-2 sentences)
- Describe a pattern/situation, NOT a solution
- Include: code structure + technical pattern + gap/issue
- NO file names, function names, or identifiers
- NO advice verbs: "should", "need to", "ensure", "avoid"

VOCABULARY GUIDANCE:
Prefer structural/technical terms over domain-specific ones.
Instead of naming WHAT the code does (authentication, payment, pagination),
describe HOW it does it (strategy pattern, numeric parameters, nested object access).

Domain terms ARE acceptable when they add clarity that technical terms cannot capture.
Example: "pagination boundary" may be clearer than "numeric range limit".

SELF-CHECK before including each query:
1. Does it describe a pattern (not a solution)?
2. Could it plausibly exist in the example database above?
3. Is it 20-50 words?
Remove any queries that fail these checks.

Generate 6-10 diverse queries covering different aspects of the diff."""

    user = f"""PR CONTEXT:
{context}

CODE DIFF:
{filtered_diff}

Generate queries that would retrieve relevant memories.
Output ONLY a JSON array of strings."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def get_sample_memories_for_prompt(db_path: str, n: int = 5) -> List[Dict[str, Any]]:
    """
    Get random sample of memories to include in query prompt.

    This grounds the LLM in actual database vocabulary.
    """
    with get_db_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT id, situation_description, lesson
            FROM memories
            ORDER BY RANDOM()
            LIMIT {n}
        """)
        return [
            {"id": row[0], "situation_description": row[1], "lesson": row[2]}
            for row in cur.fetchall()
        ]
```

### Key Changes from v2:
- **Added sample_memories parameter** - most impactful change for semantic alignment
- **Aligned length to 20-50 words** (was 10-25) to match memory length
- **Added self-verification step** - LLM filters poor outputs
- **Relaxed vocabulary rules** - allow hybrid when domain terms add clarity
- **Added prompt versioning**

---

## 3. Example Transformations

### Before/After Query Examples

**Scenario: Authentication code changes**

❌ **Current query generation:**
> "Authentication middleware replaced with passport strategy pattern. Multiple strategies registered but session configuration set per-decorator instead of globally in server settings."

✅ **Improved query (v3):**
> "Decorator applying configuration at multiple levels. Session setting duplicated between decorator parameter and server-level configuration, creating maintenance burden when values need to change."

---

**Scenario: Pagination implementation**

❌ **Current:**
> "Repository method returns paginated data with a wrapper object containing both the records array and pagination metadata. Test cases verify page boundaries but do not cover scenarios where page index exceeds available pages."

✅ **Improved:**
> "Repository method accepting numeric page parameters without bounds validation. Missing test coverage for negative values, zero page size, or index exceeding available pages."

---

**Scenario: Enum handling**

❌ **Current:**
> "New enum value added to two parallel enums in different layers. The DTO enum and the domain enum both define the same constant but lack a bidirectional mapper."

✅ **Improved:**
> "Enum extended in multiple layers without synchronization. Parallel definitions in DTO and domain layers may diverge, causing runtime type mismatch when values don't match."

---

## 4. Hybrid Query Strategy

Generate multiple query variants to improve coverage:

```python
def generate_query_variants(pattern_description: str) -> List[str]:
    """
    Generate query variants for the same pattern.

    Three perspectives help cover different phrasings in the database.
    """
    prompt = f"""Given this code pattern:
{pattern_description}

Generate 3 query variants (20-40 words each):
1. STRUCTURE-FOCUSED: Lead with code structure (test file, service, mapper)
2. PATTERN-FOCUSED: Lead with technical pattern (null check, conditional logic)
3. GAP-FOCUSED: Lead with what's missing or wrong

Return JSON array of 3 strings."""
    return call_llm(prompt)
```

Example output:
```json
[
    "Test suite for mapper method with conditional logic on nested properties. Missing test case for completely undefined parent object.",
    "Conditional null check on optional nested object. Only validates child properties when parent is defined, skips undefined parent case.",
    "Missing edge case coverage in mapper tests. Undefined parent object scenario not tested, only undefined child properties."
]
```

---

## 5. Confidence Scoring

### Use Embedding Distance (Not LLM Calls)

**Previous approach** (expensive, slow):
```python
# DON'T DO THIS - LLM call per match is expensive
def score_query_memory_match(query: str, memory: Dict) -> float:
    prompt = f"Score similarity 0-1..."
    return float(call_llm(prompt))
```

**Recommended approach** (use existing embeddings):
```python
def get_confidence_from_distance(distance: float) -> str:
    """
    Convert cosine distance to confidence label.

    Calibrated for mxbai-embed-large with cosine distance.
    """
    if distance < 0.5:
        return "high"      # Strong semantic match
    elif distance < 0.8:
        return "medium"    # Moderate match, likely relevant
    elif distance < 1.2:
        return "low"       # Weak match, may be relevant
    else:
        return "very_low"  # Poor match, likely noise


def filter_results_by_confidence(
    results: List[Dict],
    min_confidence: str = "low"
) -> List[Dict]:
    """Filter results by minimum confidence threshold."""
    confidence_order = ["very_low", "low", "medium", "high"]
    min_idx = confidence_order.index(min_confidence)

    return [
        r for r in results
        if confidence_order.index(get_confidence_from_distance(r["distance"])) >= min_idx
    ]
```

### When to Use LLM-Based Scoring

Reserve LLM scoring for:
- Offline analysis and debugging
- A/B test evaluation
- Investigating specific failures

```python
def analyze_retrieval_failure(query: str, expected_memory: Dict, retrieved: List[Dict]):
    """
    Debug why expected memory wasn't retrieved.
    Use sparingly - expensive LLM calls.
    """
    prompt = f"""
Query: {query}

Expected memory (NOT retrieved):
{expected_memory['situation_description']}

Top retrieved memory:
{retrieved[0]['situation_description'] if retrieved else 'None'}

Why might the expected memory not match the query semantically?
What vocabulary differences exist?
"""
    return call_llm(prompt)
```

---

## 6. Testing and Metrics

### A/B Test Framework

```python
def run_prompt_ab_test(
    test_cases_dir: str,
    prompt_a_fn: Callable,
    prompt_b_fn: Callable,
    n_trials: int = 3
) -> Dict:
    """
    Compare two prompt strategies with statistical significance.

    Returns metrics for both precision and recall.
    """
    results = {"a": [], "b": []}

    for tc_path in Path(test_cases_dir).glob("*.json"):
        tc = load_json(tc_path)
        ground_truth = set(tc.get("ground_truth_memory_ids", []))

        for _ in range(n_trials):
            for label, prompt_fn in [("a", prompt_a_fn), ("b", prompt_b_fn)]:
                queries = generate_queries(tc, prompt_fn)
                retrieved = run_queries(queries)
                retrieved_ids = {r["id"] for r in retrieved}

                # Calculate metrics
                true_positives = len(retrieved_ids & ground_truth)
                recall = true_positives / len(ground_truth) if ground_truth else 0
                precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                results[label].append({
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "queries_with_hits": sum(1 for q in queries if any(r["is_ground_truth"] for r in q.get("results", [])))
                })

    # Aggregate results
    return {
        "prompt_a": {
            "mean_recall": np.mean([r["recall"] for r in results["a"]]),
            "mean_precision": np.mean([r["precision"] for r in results["a"]]),
            "mean_f1": np.mean([r["f1"] for r in results["a"]]),
        },
        "prompt_b": {
            "mean_recall": np.mean([r["recall"] for r in results["b"]]),
            "mean_precision": np.mean([r["precision"] for r in results["b"]]),
            "mean_f1": np.mean([r["f1"] for r in results["b"]]),
        },
        "improvement": {
            "recall": np.mean([r["recall"] for r in results["b"]]) - np.mean([r["recall"] for r in results["a"]]),
            "precision": np.mean([r["precision"] for r in results["b"]]) - np.mean([r["precision"] for r in results["a"]]),
        },
        "p_value_recall": stats.ttest_ind(
            [r["recall"] for r in results["a"]],
            [r["recall"] for r in results["b"]]
        ).pvalue
    }
```

### Per-Query Metrics

Track individual query performance to identify patterns:

```python
def analyze_query_performance(experiment_results: Dict) -> Dict:
    """
    Analyze which queries perform well vs poorly.
    """
    query_stats = []

    for query_result in experiment_results["queries"]:
        hits = sum(1 for r in query_result["results"] if r["is_ground_truth"])
        total = len(query_result["results"])

        query_stats.append({
            "query": query_result["query"],
            "word_count": len(query_result["query"].split()),
            "hits": hits,
            "precision": hits / total if total > 0 else 0,
            "avg_distance": np.mean([r["distance"] for r in query_result["results"]]) if query_result["results"] else None,
        })

    return {
        "queries_with_hits": sum(1 for q in query_stats if q["hits"] > 0),
        "total_queries": len(query_stats),
        "query_hit_rate": sum(1 for q in query_stats if q["hits"] > 0) / len(query_stats),
        "best_queries": sorted(query_stats, key=lambda x: -x["hits"])[:3],
        "worst_queries": sorted(query_stats, key=lambda x: x["hits"])[:3],
    }
```

---

## 7. Memory Extraction Validation

Validate extracted memories before storing:

```python
def validate_memory_searchability(situation: str) -> Tuple[bool, List[str]]:
    """
    Validate that a situation description will be searchable.

    Returns (is_valid, list_of_issues)
    """
    issues = []

    # Check length (aligned with query length)
    word_count = len(situation.split())
    if word_count < 20:
        issues.append(f"Too short ({word_count} words, need 20+)")
    if word_count > 70:
        issues.append(f"Too long ({word_count} words, max 70)")

    # Check for structure terms
    structure_terms = ["test", "mapper", "service", "decorator", "method",
                       "class", "function", "file", "repository", "controller",
                       "validator", "handler", "filter", "factory"]
    if not any(term in situation.lower() for term in structure_terms):
        issues.append("Missing code structure term")

    # Check for pattern terms
    pattern_terms = ["optional", "null", "undefined", "conditional", "nested",
                     "chaining", "check", "validation", "logic", "handling",
                     "enum", "array", "object", "property", "parameter"]
    if not any(term in situation.lower() for term in pattern_terms):
        issues.append("Missing technical pattern term")

    # Check for gap terms
    gap_terms = ["missing", "lacks", "inconsistent", "redundant", "contradicts",
                 "without", "no ", "incorrect", "mismatch", "error", "duplicate",
                 "unused", "untested", "incomplete"]
    if not any(term in situation.lower() for term in gap_terms):
        issues.append("Missing gap/issue term")

    # Check for forbidden content
    if any(word in situation.lower() for word in ["should", "need to", "must", "ensure", "always"]):
        issues.append("Contains advice/solution language")

    return (len(issues) == 0, issues)
```

---

## 8. Output Parsing Robustness

Improve JSON parsing reliability:

```python
def parse_queries_robust(response: str) -> List[str]:
    """
    Robustly parse query list from LLM response.

    Handles various output formats and malformed JSON.
    """
    # Try direct JSON parse first
    try:
        # Look for JSON array
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:MAX_QUERIES_PER_EXPERIMENT]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]{20,200})"', response)
    if quoted:
        return quoted[:MAX_QUERIES_PER_EXPERIMENT]

    # Last resort: split by newlines and clean
    lines = []
    for line in response.split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*]+\s*', '', line)  # Remove list markers
        line = line.strip('"\'')
        if 20 <= len(line) <= 200:
            lines.append(line)

    return lines[:MAX_QUERIES_PER_EXPERIMENT]
```

---

## Summary: Key Changes

| Component | v2 (Previous) | v3 (Current) |
|-----------|---------------|--------------|
| **Memory length** | 40-120 words | 25-60 words |
| **Query length** | 10-25 words | 20-50 words |
| **Length alignment** | Misaligned (4x difference) | Aligned (~2x) |
| **Memory structure** | Rigid 2-sentence template | Flexible with key elements |
| **Query examples** | Static/abstract | Real memories from database |
| **Vocabulary** | No domain terms | Hybrid (prefer technical, allow domain) |
| **Self-verification** | None | Built into prompt |
| **Confidence scoring** | LLM-based (expensive) | Embedding distance (fast) |
| **Metrics** | Recall only | Precision + Recall + F1 |
| **Versioning** | None | Explicit version strings |

---

## Implementation Priority

| Priority | Change | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| **1** | Align memory/query lengths (25-60 words) | High | Low |
| **2** | Include sample memories in query prompt | High | Medium |
| **3** | Add self-verification step to query prompt | Medium | Low |
| **4** | Track precision alongside recall | Medium | Low |
| **5** | Add prompt versioning | Medium | Low |
| **6** | Relax rigid structure constraints | Medium | Low |
| **7** | Use embedding distance for confidence | Low | Low |
| **8** | Add per-query metrics | Low | Medium |
