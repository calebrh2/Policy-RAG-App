"""Fixed questions, the chunk each one should retrieve, and the gold answer.

Purpose
-------
The retrieval harness scores ``CASES`` against the current chunk that states
the answer. Generation eval scores ``GENERATION_CASES``, which adds questions
the corpus cannot answer.

Contents
--------
- ``EvalCase``: one question, its expected chunk, and its gold answer.
- ``CASES``: questions the corpus can answer.
- ``REFUSAL_CASES``: questions the corpus cannot answer.
- ``GENERATION_CASES``: ``CASES`` followed by ``REFUSAL_CASES``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.generate import REFUSAL


@dataclass(frozen=True)
class EvalCase:
    """One evaluation question.

    ``document_id``, ``section_path``, and ``needle`` identify the current
    chunk that contains the answer. They are empty when the corpus cannot
    answer. ``gold_answer`` is the reference reply. ``expect_supported`` is
    false when that reply is the refusal.
    """

    case_id: str
    question: str
    document_id: str
    section_path: str
    needle: str
    gold_answer: str
    expect_supported: bool


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="water-positive-definition",
        question='How does the Water Management Policy define "Water Positive"?',
        document_id="water-management-policy",
        section_path="Purpose",
        needle="replenishing more water than is consumed",
        gold_answer=(
            'Coforge defines "Water Positive" as replenishing more water than is consumed '
            "across owned operations."
        ),
        expect_supported=True,
    ),
    EvalCase(
        case_id="water-risk-tools",
        question="Which tools does the Water Management Policy name for water risk assessments?",
        document_id="water-management-policy",
        section_path="Water Risk Assessment",
        needle="WRI Aqueduct Water Risk Atlas",
        gold_answer="The WRI Aqueduct Water Risk Atlas or the WWF Water Risk Filter.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="water-grievance",
        question="How can someone raise a concern about water use or water quality under the Water Management Policy?",
        document_id="water-management-policy",
        section_path="Grievance Mechanism",
        needle="Whistleblower channel",
        gold_answer=(
            "Through Coforge's Whistleblower channel or the applicable local grievance "
            "redressal mechanism."
        ),
        expect_supported=True,
    ),
    EvalCase(
        case_id="water-stp-reuse",
        question="What non-potable uses does the Water Management Policy give for treated wastewater from STPs and ETPs?",
        document_id="water-management-policy",
        section_path="Water Recycling, Harvesting and Reuse",
        needle="flushing, irrigation, landscaping",
        gold_answer="Flushing, irrigation, landscaping, and HVAC/cooling systems.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="plastic-cups-prohibited",
        question="Are plastic plates, cups, and glasses prohibited within Coforge premises?",
        document_id="single-use-plastic-free-policy",
        section_path="Elimination & Substitution (Mandatory Prohibition)",
        needle="Plastic plates, cups, and glasses",
        gold_answer="Yes. Plastic plates, cups, and glasses are prohibited.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="plastic-exemption-approver",
        question="Who must give written approval for a single-use plastic exemption?",
        document_id="single-use-plastic-free-policy",
        section_path="Exemptions/Waiver",
        needle="Site Admin Head and Sustainability/ESG",
        gold_answer="The Site Admin Head and Sustainability/ESG.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="plastic-sup-deadline",
        question="By when must policy-prohibited single-use plastic items be eliminated at Coforge India sites?",
        document_id="single-use-plastic-free-policy",
        section_path="Targets & KPIs (India – All Sites) > Targets",
        needle="100% elimination of Policy-prohibited SUP items",
        gold_answer="By Dec 2026.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="carbon-reduction-target",
        question="By how much and by which year does the current Carbon Reduction Plan commit to relatively reduce emissions from the baseline year?",
        document_id="carbon-reduction-plan",
        section_path="Emissions reduction targets",
        needle="reduce them by 20% by 2030",
        gold_answer="Relatively reduce emissions by 20% by 2030 from the baseline year.",
        expect_supported=True,
    ),
    EvalCase(
        case_id="carbon-iso-standards",
        question="Which ISO standards does the current Carbon Reduction Plan say the environment, health, and safety management system conforms to?",
        document_id="carbon-reduction-plan",
        section_path="Carbon Reduction Initiatives",
        needle="ISO 14001:2015 & ISO 45001:2018",
        gold_answer="ISO 14001:2015 and ISO 45001:2018.",
        expect_supported=True,
    ),
)


REFUSAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="travel-expense-deadline",
        question="What is the deadline for travel expense reports?",
        document_id="",
        section_path="",
        needle="",
        gold_answer=REFUSAL,
        expect_supported=False,
    ),
    EvalCase(
        case_id="parental-leave-weeks",
        question="How many weeks of parental leave does the policy grant?",
        document_id="",
        section_path="",
        needle="",
        gold_answer=REFUSAL,
        expect_supported=False,
    ),
)


GENERATION_CASES: tuple[EvalCase, ...] = CASES + REFUSAL_CASES
