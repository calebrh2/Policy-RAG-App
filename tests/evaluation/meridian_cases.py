"""Meridian questions, the chunk each one should retrieve, and the gold answer.

Purpose
-------
Retrieval scores ``MERIDIAN_CASES`` against the current chunk that states the
answer. Generation eval scores ``MERIDIAN_GENERATION_CASES``, which adds
questions this corpus cannot answer. The outdated leave edition is not a
gold chunk: needles name rules that exist only in the March 2026 text.

Contents
--------
- ``MERIDIAN_CASES``: questions the corpus can answer.
- ``MERIDIAN_REFUSAL_CASES``: questions the corpus cannot answer.
- ``MERIDIAN_GENERATION_CASES``: answerable questions followed by refusals.
"""

from __future__ import annotations

from rag.generate import REFUSAL
from tests.evaluation.cases import EvalCase

_LEAVE = "meridian-analytics-leave-policy"
_HEALTH = "meridian-analytics-health-benefits-policy"
_SECURITY = "meridian-analytics-it-data-security-policy"

MERIDIAN_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="leave-pto-days",
        question="How many days of Paid Time Off does the current Leave Policy accrue per year?",
        document_id=_LEAVE,
        section_path="3. Vacation / Paid Time Off",
        needle="20 days per year",
        gold_answer="Employees accrue 20 days of Paid Time Off per year.",
        key_facts=("20 days",),
        expect_supported=True,
    ),
    EvalCase(
        case_id="leave-sick-days",
        question="How many sick leave days does the current Leave Policy give per year?",
        document_id=_LEAVE,
        section_path="4. Sick Leave",
        needle="10 days of sick leave",
        gold_answer="Employees receive 10 days of sick leave per year.",
        key_facts=("10 days",),
        expect_supported=True,
    ),
    EvalCase(
        case_id="leave-parental-weeks",
        question=(
            "How many weeks of parental leave does the current Leave Policy grant, "
            "and how is that leave paid?"
        ),
        document_id=_LEAVE,
        section_path="5. Parental Leave",
        needle="up to 12 weeks",
        gold_answer=(
            "Up to 12 weeks. The first six weeks are paid in full, and the remaining "
            "six weeks are paid at 50% of base salary."
        ),
        key_facts=("12 weeks", "50%"),
        expect_supported=True,
    ),
    EvalCase(
        case_id="health-hsa-contribution",
        question=(
            "How much does the Health & Benefits Policy contribute each year to an HSA "
            "for employees who elect the HDHP?"
        ),
        document_id=_HEALTH,
        section_path="3. Medical, Dental, and Vision Coverage",
        needle="$750 for employee-only coverage or $1,500 for family coverage",
        gold_answer="$750 for employee-only coverage or $1,500 for family coverage.",
        key_facts=("$750", "$1,500"),
        expect_supported=True,
    ),
    EvalCase(
        case_id="health-enrollment-window",
        question="How long do new employees have to enroll in benefits under the Health & Benefits Policy?",
        document_id=_HEALTH,
        section_path="4. Enrollment Periods",
        needle="within 30 days of their eligibility date",
        gold_answer="30 days from their eligibility date.",
        key_facts=("30 days",),
        expect_supported=True,
    ),
    EvalCase(
        case_id="health-wellness-reimbursement",
        question="What is the annual wellness reimbursement under the Health & Benefits Policy?",
        document_id=_HEALTH,
        section_path="5. Wellness Program",
        needle="up to $300 per year",
        gold_answer="Up to $300 per year.",
        key_facts=("$300",),
        expect_supported=True,
    ),
    EvalCase(
        case_id="security-mfa",
        question="Does the IT & Data Security Policy require multi-factor authentication for company accounts?",
        document_id=_SECURITY,
        section_path="3. Device and Account Security",
        needle="Multi-factor authentication (MFA) is required",
        gold_answer="Yes. Multi-factor authentication is required for all company accounts.",
        key_facts=("Multi-factor authentication",),
        expect_supported=True,
    ),
    EvalCase(
        case_id="security-incident-hour",
        question="How soon must a suspected security incident be reported under the IT & Data Security Policy?",
        document_id=_SECURITY,
        section_path="6. Incident Reporting",
        needle="within 1 hour of discovery",
        gold_answer="Within 1 hour of discovery.",
        key_facts=("1 hour",),
        expect_supported=True,
    ),
)

MERIDIAN_REFUSAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="travel-expense-deadline",
        question="What is the deadline for travel expense reports?",
        document_id="",
        section_path="",
        needle="",
        gold_answer=REFUSAL,
        key_facts=(REFUSAL,),
        expect_supported=False,
    ),
    EvalCase(
        case_id="plastic-cups-prohibited",
        question="Are plastic plates, cups, and glasses prohibited?",
        document_id="",
        section_path="",
        needle="",
        gold_answer=REFUSAL,
        key_facts=(REFUSAL,),
        expect_supported=False,
    ),
)

MERIDIAN_GENERATION_CASES: tuple[EvalCase, ...] = MERIDIAN_CASES + MERIDIAN_REFUSAL_CASES
