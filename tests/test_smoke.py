import dissonant

import dynamic_tuning


def test_imports() -> None:
    assert dynamic_tuning.__doc__
    assert dissonant.model_by_name("sethares1993") is not None
