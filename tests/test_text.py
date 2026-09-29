from src.text import normalize_am, normalize_en


def test_english_lowercased_and_quotes_unified():
    assert normalize_en("  “Hello”   World  ") == '"hello" world'


def test_verse_reference_removed():
    assert normalize_en("woes to scribes (13 36)") == "woes to scribes"


def test_amharic_homophones_collapsed():
    # ሐ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ families (all vowel orders)
    assert normalize_am("ሐሙስ ሠላም ዓለም ፀሐይ") == "ሀሙስ ሰላም ኣለም ጸሀይ"


def test_amharic_punctuation():
    assert normalize_am("ሰላም ነው ።") == "ሰላም ነው።"
    assert normalize_am("ሰላም፡ነው::") == "ሰላም ነው።"
