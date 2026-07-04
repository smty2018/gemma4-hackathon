from app.agent.orchestrator import build_analysis_plan
from app.schemas.document import AnalysisStage


def test_analysis_plan_requires_confirmation() -> None:
    plan = build_analysis_plan(
        filename="bill.pdf",
        content_type="application/pdf",
        language="bn",
        audience="grandmother",
    )

    assert plan.status is AnalysisStage.ACCEPTED
    assert plan.requires_confirmation_before_actions is True
    assert len(plan.steps) == 3
