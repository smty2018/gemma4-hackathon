import argparse
from pathlib import Path

from app.demo import DemoFlow, DemoFlowRequest
from app.inference import GemmaAdapter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gemma 4 document demo flow on a UTF-8 text sample."
    )
    parser.add_argument("document", type=Path)
    parser.add_argument("--language", default="English")
    parser.add_argument("--audience", default="general public")
    parser.add_argument("--style", default="Simple")
    parser.add_argument(
        "--question",
        default="What should I do and by when?",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    document_text = arguments.document.read_text(encoding="utf-8")
    result = DemoFlow(GemmaAdapter()).run(
        DemoFlowRequest(
            document_text=document_text,
            language=arguments.language,
            audience=arguments.audience,
            explanation_style=arguments.style,
            follow_up_question=arguments.question,
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
