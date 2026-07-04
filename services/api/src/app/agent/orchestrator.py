from app.schemas.document import AnalysisPlan, AnalysisStage, PlannedStep


def build_analysis_plan(
    *, filename: str, content_type: str, language: str, audience: str
) -> AnalysisPlan:
    return AnalysisPlan(
        document_name=filename,
        content_type=content_type,
        language=language,
        audience=audience,
        steps=[
            PlannedStep(
                stage=AnalysisStage.EXTRACT_EVIDENCE,
                description="Render pages and extract evidence with Gemma multimodal inference.",
            ),
            PlannedStep(
                stage=AnalysisStage.VERIFY_FACTS,
                description="Validate dates, amounts, and source spans before explanation.",
            ),
            PlannedStep(
                stage=AnalysisStage.BUILD_ACTION_PLAN,
                description="Prepare grounded next steps and confirmation-gated tool proposals.",
            ),
        ],
    )
