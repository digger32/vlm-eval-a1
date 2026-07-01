"""Answer parsing.

Two regimes:
  - MCQ  : map free-text model output -> chosen option letter (A/B/C/...).
  - open : VQA-v2 / VizWiz style soft accuracy against 10 human answers.

Both are deliberately conservative: we want to UNDER-credit clever guessing, not
over-credit it, since the papers critique guessing.
"""
from __future__ import annotations
import re
import string

# ---------------------------------------------------------------- MCQ parsing

_LETTER_RE = re.compile(r"\b([A-H])\b")


def parse_mcq(output: str, options: list[str]) -> str | None:
    """Return the chosen option LETTER ('A'..) or None if unparseable.

    Strategy, in order:
      1. explicit leading letter ("A", "A.", "(A)", "A)") at start of output;
      2. any standalone capital letter within range;
      3. exact/substring match of the output against an option's text.
    """
    if not output:
        return None
    n = len(options)
    valid = set(string.ascii_uppercase[:n])
    s = output.strip()

    # 1. leading letter forms
    m = re.match(r"\(?\s*([A-H])\s*[\).:]", s)
    if m and m.group(1) in valid:
        return m.group(1)
    if len(s) >= 1 and s[0].upper() in valid and (len(s) == 1 or not s[1].isalpha()):
        return s[0].upper()

    # 2. any standalone letter in valid range (first occurrence)
    for m in _LETTER_RE.finditer(s.upper()):
        if m.group(1) in valid:
            return m.group(1)

    # 3. text match against option strings
    s_low = s.lower()
    for i, opt in enumerate(options):
        if opt and opt.strip().lower() in s_low:
            return string.ascii_uppercase[i]
    return None


# ----------------------------------------------------- open-ended VQA accuracy

_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "dont": "don't", "isnt": "isn't", "arent": "aren't", "cant": "can't",
    "wont": "won't", "didnt": "didn't", "doesnt": "doesn't",
}
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _PUNCT.sub("", ans)
    toks = [t for t in ans.split() if t not in _ARTICLES]
    toks = [_CONTRACTIONS.get(t, t) for t in toks]
    return " ".join(toks)


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    """Standard VQA soft accuracy: min(#matches/3, 1), averaged over 10 leave-one-out
    human-answer subsets (here approximated by the standard closed form)."""
    p = _normalize(pred)
    gts = [_normalize(a) for a in gt_answers]
    matches = sum(1 for g in gts if g == p)
    return min(matches / 3.0, 1.0)


def is_unanswerable(pred: str) -> bool:
    """VizWiz includes 'unanswerable'. Treat abstention-like outputs explicitly so we
    can separate 'wrong' from 'declined' (relevant to A1's blind-LLM control)."""
    p = _normalize(pred)
    return p in {"unanswerable", "unsuitable", "cannot tell", "i dont know",
                 "unknown", "not sure", "cant tell"}
