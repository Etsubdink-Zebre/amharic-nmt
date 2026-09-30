from src.translator import apply_suggestions, edit_distance, load_word_freq, suggest_spellings

FREQ = load_word_freq()


def test_edit_distance_counts_a_swap_as_one_typo():
    assert edit_distance("thnak", "thank") == 1
    assert edit_distance("hospitl", "hospital") == 1


def test_common_typos_are_corrected():
    words = ["switherland", "hospitl", "thnak", "goverment", "univercity"]
    assert suggest_spellings(words, FREQ) == {
        "switherland": "switzerland", "hospitl": "hospital", "thnak": "thank",
        "goverment": "government", "univercity": "university"}


def test_known_and_unrelated_words_get_no_suggestion():
    assert suggest_spellings(["switzerland", "chromodynamics"], FREQ) == {}


def test_apply_suggestions_keeps_capitals():
    assert apply_suggestions("Thnak you, hospitl.", {"thnak": "thank", "hospitl": "hospital"}) == "Thank you, hospital."
