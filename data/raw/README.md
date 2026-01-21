# Raw Data Directory

Contains code review data collected from repositories. Files are gitignored (sensitive).

## File Naming Convention

```
<type>-<ticket>.json
```

Examples:
- `feature-JIRA-1234.json`
- `bugfix-JIRA-5679.json`

## JSON Schema

```json
{
  "context": "string - PR description and review context summary",
  "meta": {
    "sourceBranch": "feature/JIRA-XXX",
    "targetBranch": "release/vX.Y.Z",
    "gatheredAt": "ISO 8601 timestamp",
    "gatheredFromCommit": "commit SHA",
    "repoPath": "local path to repository",
    "repoRemote": "git remote URL",
    "version": "1.0"
  },
  "code_review_comments": [
    {
      "id": "UUID",
      "file": "path/to/file.ts",
      "line": 123,
      "severity": "issue | suggestion | risk",
      "message": "The review comment text",
      "rationale": "Why this matters (optional)",
      "confidence": "high | medium | low",
      "verifiedBy": "How the issue was verified",
      "status": "accepted | rejected",
      "code_snippet": "Relevant code fragment"
    }
  ],
  "full_diff": "string - Complete PR diff in unified format",
  "reviewedAt": "ISO 8601 timestamp"
}
```
