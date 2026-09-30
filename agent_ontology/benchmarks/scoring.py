#!/usr/bin/env python3
"""
Scoring functions for benchmark evaluation.

Provides:
  - exact_match: normalized string comparison
  - f1_score: token-level precision/recall
  - exact_match_number: extract final number and compare
  - normalize_answer: lowercase, strip articles/punctuation/whitespace
  - extract_final_number: regex patterns for #### N, "answer is N", last number
  - extract_answer: priority-key search over state.data for final answer
"""

import re
import string


def normalize_answer(s):
    """Lowercase, strip articles/punctuation/extra whitespace."""
    s = str(s).lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(predicted, expected):
    """Normalized string comparison. Returns 0.0 or 1.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(expected) else 0.0


def f1_score(predicted, expected):
    """Token-level F1 score. Returns 0.0–1.0."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(expected).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    # Count occurrences
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    num_common = sum(min(pred_counts[t], gold_counts[t]) for t in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_final_number(text):
    """Extract the final numeric answer from text.

    Tries patterns in order:
      1. #### N  (GSM8K standard format)
      2. "the answer is N" / "answer: N"
      3. Last number in the text
    Returns the number as a float, or None if not found.
    """
    text = str(text)

    # Pattern 1: #### N (GSM8K format)
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 2: "the answer is N" or "answer: N" or "answer = N"
    m = re.search(
        r'(?:the\s+)?answer\s*(?:is|:|=)\s*(-?[\d,]+\.?\d*)',
        text, re.IGNORECASE
    )
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 3: "= N" at end of a line (common in math solutions)
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)\s*$', text, re.MULTILINE)
    if m:
        return float(m.group(1).replace(',', ''))

    # Pattern 4: Last standalone number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def exact_match_number(predicted, expected_number):
    """Extract final number from predicted text, compare to expected.
    Returns 0.0 or 1.0."""
    pred_num = extract_final_number(predicted)
    if pred_num is None:
        return 0.0
    try:
        exp = float(expected_number)
    except (ValueError, TypeError):
        return 0.0
    # Compare with tolerance for floating point
    if abs(exp) < 1e-9:
        return 1.0 if abs(pred_num) < 1e-9 else 0.0
    return 1.0 if abs(pred_num - exp) / max(abs(exp), 1e-9) < 1e-6 else 0.0


def extract_answer(state_data, inputs=None):
    """Extract the final answer from agent state data.

    Searches keys in priority order:
      answer, final_answer, final_output, output_text, result,
      current_output, synthesized_answer, response
    Also searches nested dicts (e.g. state.data["xxx_result"]["answer"]).
    """
    if not isinstance(state_data, dict):
        return str(state_data)

    priority_keys = [
        "answer", "final_answer", "final_output", "output_text",
        "result", "current_output", "synthesized_answer", "response",
    ]

    # Direct lookup
    for key in priority_keys:
        val = state_data.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            return val.strip()

    # Search nested dicts
    for k, v in state_data.items():
        if isinstance(v, dict):
            for key in priority_keys:
                val = v.get(key)
                if val and isinstance(val, str) and len(val.strip()) > 0:
                    return val.strip()

    # Fallback: concatenate the string values the agent produced. `inputs` are
    # what the benchmark passed in; without excluding them the question text was
    # scored as the answer (an ARC question's "A) ..." read as choosing A).
    inputs = inputs or {}
    input_values = {str(v).strip() for v in inputs.values() if isinstance(v, str)}
    parts = []
    for k, v in state_data.items():
        if k in inputs or not isinstance(v, str) or not v.strip() or v.strip() in input_values:
            continue
        parts.append(v.strip())
    return ' '.join(parts) if parts else ""


def score_hotpotqa(predicted, expected):
    """Score a HotpotQA prediction. Returns dict with em and f1."""
    return {
        "em": exact_match(predicted, expected),
        "f1": f1_score(predicted, expected),
    }


def score_gsm8k(predicted, expected_number):
    """Score a GSM8K prediction. Returns dict with em."""
    return {
        "em": exact_match_number(predicted, expected_number),
    }


def extract_choice_letter(text):
    """Extract a multiple-choice answer letter (A-D) from text.

    Tries patterns in order:
      1. "the answer is X" / "answer: X"
      2. Standalone letter at end of text
      3. First letter followed by ) in the text
      4. First standalone A/B/C/D
    """
    text = str(text).strip()

    # Pattern 1: "the answer is X" / "answer: X" / "answer is X"
    m = re.search(r'(?:the\s+)?answer\s*(?:is|:|=)\s*([A-Da-d])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: Standalone letter at end
    m = re.search(r'\b([A-Da-d])\s*[.)]*\s*$', text)
    if m:
        return m.group(1).upper()

    # Pattern 3: "X)" pattern (choice format)
    m = re.search(r'\b([A-Da-d])\)', text)
    if m:
        return m.group(1).upper()

    # Pattern 4: First standalone A/B/C/D
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)

    return None


def exact_match_choice(predicted, expected_letter):
    """Extract choice letter from predicted text, compare to expected.
    Returns 0.0 or 1.0."""
    pred_letter = extract_choice_letter(predicted)
    if pred_letter is None:
        return 0.0
    return 1.0 if pred_letter.upper() == str(expected_letter).upper() else 0.0


def score_arc(predicted, expected_letter):
    """Score an ARC prediction. Returns dict with em."""
    return {
        "em": exact_match_choice(predicted, expected_letter),
    }


def extract_code(text):
    """Extract Python code from agent output.

    Tries patterns:
      1. Code block (```python ... ```)
      2. Code block (``` ... ```)
      3. def function_name(...): pattern
      4. Raw text as code
    """
    text = str(text).strip()

    # Pattern 1: ```python block
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Pattern 2: ``` block
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Pattern 3: starts with def
    m = re.search(r'(def\s+\w+\(.*?\).*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return text


def execute_code_with_tests(code_str, test_cases_str, timeout=5):
    """Execute generated code and run test cases.

    Returns dict with:
      - passed: bool
      - error: str or None
      - tests_run: int
      - tests_passed: int
    """
    import subprocess
    import tempfile
    import os

    # Build the test script
    script = f"{code_str}\n\n{test_cases_str}\nprint('ALL_TESTS_PASSED')\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['python3', tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        passed = 'ALL_TESTS_PASSED' in result.stdout
        error = result.stderr.strip() if result.returncode != 0 else None
        # Count tests
        tests = [l for l in test_cases_str.strip().split('\n') if l.strip().startswith('assert')]
        return {
            "passed": passed,
            "error": error,
            "tests_run": len(tests),
            "tests_passed": len(tests) if passed else 0,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout", "tests_run": 0, "tests_passed": 0}
    except Exception as e:
        return {"passed": False, "error": str(e), "tests_run": 0, "tests_passed": 0}
    finally:
        os.unlink(tmp_path)


def score_multidoc(predicted, expected):
    """Score a MultiDoc prediction. Returns dict with em and f1.

    Uses flexible matching: tries exact match first, then numeric match,
    then containment (expected answer found within predicted text).
    """
    em = exact_match(predicted, expected)
    if em == 1.0:
        return {"em": 1.0, "f1": 1.0}

    # Try numeric match (handles "1200" vs "1,200 participants" and "$10.2 million" vs "10200000")
    def _parse_number_with_multiplier(s):
        """Parse numbers with optional multipliers (million, billion, etc.)."""
        s_clean = re.sub(r'[,$]', '', str(s)).strip()
        multipliers = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9, 'trillion': 1e12}
        for word, mult in multipliers.items():
            if word in s_clean.lower():
                num_part = re.search(r'[\d.]+', s_clean)
                if num_part:
                    return float(num_part.group()) * mult
        try:
            return float(re.sub(r'[^\d.\-]', '', s_clean))
        except (ValueError, TypeError):
            return None

    exp_num = _parse_number_with_multiplier(expected)
    pred_num = _parse_number_with_multiplier(predicted)
    if exp_num is not None and pred_num is not None:
        if abs(exp_num) < 1e-9:
            if abs(pred_num) < 1e-9:
                return {"em": 1.0, "f1": 1.0}
        elif abs(pred_num - exp_num) / max(abs(exp_num), 1e-9) < 0.001:
            return {"em": 1.0, "f1": 1.0}

    # Try bidirectional containment (handles "MIT" vs "...MIT..." and "8.4 miles" vs "8.4")
    norm_pred = normalize_answer(predicted)
    norm_exp = normalize_answer(expected)
    if norm_exp and norm_pred and (norm_exp in norm_pred or norm_pred in norm_exp):
        return {"em": 1.0, "f1": 1.0}

    return {"em": 0.0, "f1": f1_score(predicted, expected)}


def score_kb_tool(predicted, expected):
    """Score a KB tool prediction. Returns dict with em and f1.

    Uses flexible matching: exact match, then numeric, then containment, then F1.
    Same logic as score_multidoc — works for both string and numeric answers.
    """
    em = exact_match(predicted, expected)
    if em == 1.0:
        return {"em": 1.0, "f1": 1.0}

    # Try numeric match
    try:
        exp_num = float(re.sub(r'[,$]', '', str(expected)))
        pred_num = float(re.sub(r'[,$]', '', str(predicted)))
        if abs(exp_num) < 1e-9:
            if abs(pred_num) < 1e-9:
                return {"em": 1.0, "f1": 1.0}
        elif abs(pred_num - exp_num) / max(abs(exp_num), 1e-9) < 0.001:
            return {"em": 1.0, "f1": 1.0}
    except (ValueError, TypeError):
        pass

    # Try bidirectional containment
    norm_pred = normalize_answer(predicted)
    norm_exp = normalize_answer(expected)
    if norm_exp and norm_pred and (norm_exp in norm_pred or norm_pred in norm_exp):
        return {"em": 1.0, "f1": 1.0}

    # Try word-set containment (all expected words appear in predicted)
    if norm_exp and norm_pred:
        exp_words = set(norm_exp.split())
        pred_words = set(norm_pred.split())
        if exp_words and exp_words.issubset(pred_words):
            return {"em": 1.0, "f1": 1.0}

    return {"em": 0.0, "f1": f1_score(predicted, expected)}


def score_humaneval(predicted, example):
    """Score a HumanEval prediction. Returns dict with pass_at_1."""
    code = extract_code(predicted)
    test_cases = example.get("test_cases", "")
    if not test_cases:
        return {"pass_at_1": 0.0}
    result = execute_code_with_tests(code, test_cases)
    return {
        "pass_at_1": 1.0 if result["passed"] else 0.0,
    }


def score_meta_improve(predicted, example):
    """Score a meta_improve prediction. Delegates to meta_eval module.

    Returns dict with em (0.0 or 1.0) and eval_details.
    """
    from .meta_eval import score_meta_improve as _score_meta_improve
    return _score_meta_improve(predicted, example)
