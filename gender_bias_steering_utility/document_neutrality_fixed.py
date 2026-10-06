"""Punctuation-stripped tokenization for DocumentNeutrality, without touching the original.

document_neutrality.py's DocumentNeutrality class has no tokenizer of its own --
get_magnitude_count()/get_neutrality() just take whatever token list the caller hands
them. calc_documents_neutrality.py's own script tokenizes with doctext.lower().split(' '),
leaving punctuation attached to tokens ("boy,", "husband.", "he's"), which then never
exact-matches the wordlist. diagnose_tokenization_undercounting.py confirmed this causes
real undercounting (some passages that obviously mention gender score as fully neutral).

This subclass changes only the tokenization step; get_magnitude_count() and
get_neutrality() are inherited unmodified, so the scoring logic itself is untouched.

This is the tokenization every later neutrality computation should use -- the 5
dense-model baselines and every steered condition -- import
PunctuationStrippedDocumentNeutrality from here rather than reimplementing it.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEUTRALITY_CODE_DIR = REPO_ROOT / "data/FairnessRetrievalResults/adversarial_mitigation/fairness_measurement"
sys.path.insert(0, str(NEUTRALITY_CODE_DIR))
from document_neutrality import DocumentNeutrality  # noqa: E402  (unmodified; only its scoring is reused below)

# Non-alphanumeric characters become whitespace, not nothing, so e.g. "he's" -> "he s" (two
# real tokens, "he" survives) rather than "hes" (one token that matches nothing in the wordlist).
STRIP_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


class PunctuationStrippedDocumentNeutrality(DocumentNeutrality):
    """Same representative_words / groups_portion / threshold, and the same
    get_magnitude_count() / get_neutrality() scoring (both inherited unmodified as-is)
    -- only how raw doc text becomes tokens is different."""

    @staticmethod
    def tokenize(doctext):
        return STRIP_PUNCT_RE.sub(" ", doctext.lower()).split()

    def get_neutrality_from_text(self, doctext):
        return self.get_neutrality(self.tokenize(doctext))
