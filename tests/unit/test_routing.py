from rag.retrieval import RetrievalHit
from rag.routing import (
    comparison_hits,
    family_name,
    resolve_route,
    route,
    should_compare_steps,
    version_terms,
    versioned_hits,
)


def test_normal_question_uses_the_current_plan() -> None:
    assert route("What was India Scope 2 in 2023?") == "current"


def test_word_list_does_not_call_the_model() -> None:
    model = _RouteModel("compare")

    assert route("What changed in the latest carbon plan from before?", model) == "compare"
    assert model.calls == 0


def test_differ_compares_without_calling_the_model() -> None:
    model = _RouteModel("current")

    assert route("How does the March 2024 plan differ from the September 2022 plan?", model) == "compare"
    assert model.calls == 0


def test_model_routes_a_question_the_word_list_misses() -> None:
    question = "How does the March 2024 plan stand beside the September 2022 plan?"

    assert route(question) == "current"
    assert route(question, _RouteModel("compare")) == "compare"


def test_unknown_model_reply_stays_on_the_current_plan() -> None:
    question = "How does the March 2024 plan stand beside the September 2022 plan?"

    assert route(question, _RouteModel("both plans")) == "current"


def test_current_and_outdated_steps_are_one_comparison() -> None:
    assert should_compare_steps(
        ["current", "outdated"],
        ["What does the March 2024 plan say?", "What does the September 2022 plan say?"],
        single_version=frozenset({"plastic", "water"}),
    )
    assert not should_compare_steps(
        ["current", "outdated"],
        ["What is the carbon target?", "When is the water policy water positive?"],
        single_version=frozenset({"plastic", "water"}),
    )


def test_compare_keeps_only_versioned_families() -> None:
    plastic = _hit("plastic", "current", "Targets", "Dec 2026")
    kept = versioned_hits(
        [
            _hit("carbon", "current", "Targets", "20% by 2030"),
            RetrievalHit(
                chunk_id=plastic.chunk_id,
                document_name="Single-use-Plastic-free-Policy",
                version=plastic.version,
                section=plastic.section,
                source_pages=plastic.source_pages,
                text=plastic.text,
                score=plastic.score,
            ),
        ],
        frozenset({"carbon"}),
    )

    assert [hit.chunk_id for hit in kept] == ["carbon"]


def test_outdated_question_uses_the_old_plan() -> None:
    assert route("What did the outdated plan say about Scope 2?") == "outdated"


def test_change_question_compares_both_plans() -> None:
    assert route("What changed in the latest carbon plan from before?") == "compare"


def test_a_one_version_family_is_not_compared() -> None:
    multi, single = version_terms(
        [
            ("Carbon-Reduction-Plan", "current"),
            ("Outdated-Carbon-Reduction-Plan-TEST-VERSION", "outdated"),
            ("Single-use-Plastic-free-Policy copy", "current"),
            ("Water-Management-Policy", "current"),
        ]
    )

    assert family_name("Outdated-Carbon-Reduction-Plan-TEST-VERSION") == "carbon-reduction-plan"
    assert "carbon" in multi
    assert "plastic" in single
    assert resolve_route("When are plastics eliminated?", _RouteModel("compare"), multi_version=multi, single_version=single) == "current"
    assert resolve_route("What changed in the carbon plan from before?", None, multi_version=multi, single_version=single) == "compare"


def test_comparison_pairs_a_section_and_marks_a_missing_side() -> None:
    hits = comparison_hits(
        [
            _hit("current-target", "current", "Emissions reduction targets", "reduce them by 20% by 2030"),
            _hit("current-only", "current", "New initiative", "RE100"),
        ],
        [
            _hit("old-target", "outdated", "Emissions reduction targets", "reduce them by 15% by 2027"),
        ],
    )

    assert hits[0].section == "Emissions reduction targets"
    assert "20%" in hits[0].text
    assert "15%" in hits[0].text
    assert hits[1].section == "New initiative"
    assert "not in the outdated plan" in hits[1].text


class _RouteModel:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        assert "one word only" in prompt
        return self._reply


def _hit(chunk_id: str, version: str, section: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_name="Carbon-Reduction-Plan",
        version=version,
        section=section,
        source_pages="2-2",
        text=text,
        score=0.1,
    )
