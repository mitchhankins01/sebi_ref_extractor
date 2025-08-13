import os
import sys
import pytest

# Ensure project root is on sys.path for direct module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import normalize_circular_id, normalize_date_mdy, is_gazette_code, is_valid_llm_reference


def test_normalize_circular_id_zero_padded():
    raw = "SEBI/HO/MRD2/MRD2_DCAP/P/CIR/2021/0000000591"
    assert normalize_circular_id(raw).endswith("/591")


def test_normalize_circular_id_zero():
    raw = "SEBI/HO/MRD2/MRD2_DCAP/P/CIR/2021/0"
    assert normalize_circular_id(raw).endswith("/0")


def test_normalize_date_mdy():
    assert normalize_date_mdy("dated July 5, 2021") == "2021-07-05"


@pytest.mark.parametrize(
    "text",
    [
        "S.O. 5401(E)",
        "G.S.R. 123(E)",
    ],
)
def test_is_gazette_code(text):
    assert is_gazette_code(text) is True


def test_llm_placeholder_circular_is_filtered():
    ref_text = "SEBI Circular No."
    snippet = "... as per SEBI Circular No. and other provisions ..."
    assert is_valid_llm_reference(ref_text, "circular", snippet) is False


