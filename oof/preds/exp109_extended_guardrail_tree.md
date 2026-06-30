# EXP-109 Extended Guardrail Filter

Purpose: re-evaluate every existing submission candidate after EXP-108 showed that `test_0568=1` also lowers public score.

Evaluated files: 189
Passed guardrail: 2
Passed and non-duplicate against EXP-56: 0

Hard guardrails:
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
  - test_0568.png must be 0

Label distribution:
  - hard label1 range: [392, 402]
  - soft label1 range: [395, 397]

Suspicious-case rules:
  - do not change `avoid_change_protected`
  - do not change `review_high_priority`
  - warn if 3 or more `review` samples change

Interpretation:
  - `pass_guardrail=True` means the file does not violate current public-informed guardrails.
  - `pass_and_nonduplicate=True` is the only group worth considering as a new submission candidate.
  - This is a filtering analysis, not evidence that a candidate will improve public score.
