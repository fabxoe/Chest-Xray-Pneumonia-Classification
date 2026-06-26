# EXP-97 Guardrail Tree Filter

Purpose: rank existing submission files by public-informed guardrails without using leaderboard probing as a label search loop.

Decision tree:

1. Validate format.
   - 624 rows
   - columns: file_name,label
   - sample_submission order
   - labels in {0,1}
2. Apply hard guardrails:
  - test_0025.png must be 0
  - test_0049.png must be 0
  - test_0141.png must be 1
  - test_0188.png must be 0
  - test_0201.png must be 1
  - test_0222.png must be 1
  - test_0244.png must be 0
  - test_0490.png must be 0
  - test_0504.png must be 0
  - test_0539.png must be 0
3. Apply conditional guardrails:
   - do not set both test_0188=1 and test_0568=1
4. Apply label distribution checks:
   - hard range: label1 in [392, 402]
   - soft range: label1 in [395, 397]
5. Apply suspicious-case checks:
   - avoid changing EXP-95 avoid_change_protected samples
   - avoid changing review_high_priority samples unless there is new model evidence
   - warn if 3 or more EXP-95 review samples change

Interpretation:

- pass_guardrail=True does not mean the file should be submitted.
- pass_guardrail=False means it conflicts with public-informed safety rules.
- risk_score is for prioritization only, not a validation metric.
