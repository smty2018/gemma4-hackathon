from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaAdapter,
    GemmaInferenceError,
    GemmaInputError,
    GemmaLoadError,
    GemmaRequest,
    GemmaResponseError,
)


class RecordingRuntime:
    def __init__(self, response: str = "A clear answer") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        if self.error:
            raise self.error
        return self.response


class ExtractedAmount(BaseModel):
    amount: float = Field(gt=0)
    currency: str


def adapter_for(runtime: RecordingRuntime) -> GemmaAdapter:
    return GemmaAdapter(runtime_factory=lambda _model_id: runtime)


def test_text_generation_uses_e4b_by_default() -> None:
    runtime = RecordingRuntime("Pay before Friday.")
    adapter = adapter_for(runtime)

    response = adapter.generate(
        GemmaRequest(
            prompt="Explain this notice.",
            max_new_tokens=128,
            temperature=0.3,
        )
    )

    assert response.text == "Pay before Friday."
    assert response.model_id == DEFAULT_MODEL_ID == "google/gemma-3n-E4B-it"
    assert response.structured is None
    assert runtime.calls[0]["max_new_tokens"] == 128
    assert runtime.calls[0]["temperature"] == 0.3
    assert runtime.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Explain this notice."}],
        }
    ]


def test_one_request_supports_images_audio_and_text() -> None:
    runtime = RecordingRuntime()
    image_object = object()
    audio_object = object()
    adapter = adapter_for(runtime)

    adapter.generate(
        GemmaRequest(
            prompt="Describe and transcribe these inputs.",
            images=(
                "https://example.test/notice.png",
                Path("local-scan.png"),
                image_object,
            ),
            audio=(audio_object,),
        )
    )

    content = runtime.calls[0]["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "url": "https://example.test/notice.png",
    }
    assert content[1] == {"type": "image", "path": "local-scan.png"}
    assert content[2] == {"type": "image", "image": image_object}
    assert content[3] == {"type": "audio", "audio": audio_object}
    assert content[4]["type"] == "text"


def test_system_instruction_uses_the_supported_system_role() -> None:
    runtime = RecordingRuntime()
    adapter = adapter_for(runtime)

    adapter.generate(
        GemmaRequest(
            prompt="Explain this.",
            system_instruction="Use plain language.",
        )
    )

    assert runtime.calls[0]["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Use plain language."}],
    }
    assert runtime.calls[0]["messages"][1]["role"] == "user"


def test_pydantic_schema_is_prompted_parsed_and_validated() -> None:
    runtime = RecordingRuntime('{"amount":1250,"currency":"INR"}')
    adapter = adapter_for(runtime)

    response = adapter.generate(
        GemmaRequest(
            prompt="Extract the amount.",
            response_schema=ExtractedAmount,
        )
    )

    assert response.structured == {"amount": 1250.0, "currency": "INR"}
    prompt = runtime.calls[0]["messages"][0]["content"][-1]["text"]
    assert "Return only one valid JSON object" in prompt
    assert '"currency"' in prompt


def test_json_schema_accepts_fenced_json_but_still_validates_it() -> None:
    runtime = RecordingRuntime('```json\n{"answer":"yes"}\n```')
    adapter = adapter_for(runtime)
    schema = {
        "type": "object",
        "properties": {"answer": {"enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    response = adapter.generate(
        GemmaRequest(prompt="Answer.", response_schema=schema)
    )

    assert response.structured == {"answer": "yes"}


@pytest.mark.parametrize(
    ("gemma_request", "code"),
    [
        (GemmaRequest(), "empty_gemma_request"),
        (
            GemmaRequest(prompt="hello", images=(None,)),
            "invalid_media_source",
        ),
        (
            GemmaRequest(prompt="hello", audio=(object(), object())),
            "too_many_audio_inputs",
        ),
        (
            GemmaRequest(prompt="hello", max_new_tokens=0),
            "invalid_generation_limit",
        ),
        (
            GemmaRequest(prompt="hello", max_new_tokens=4_097),
            "invalid_generation_limit",
        ),
        (
            GemmaRequest(prompt="hello", temperature=2.1),
            "invalid_temperature",
        ),
        (
            GemmaRequest(prompt="hello", response_schema={"type": "unknown"}),
            "invalid_response_schema",
        ),
    ],
)
def test_invalid_requests_have_stable_error_codes(
    gemma_request: GemmaRequest,
    code: str,
) -> None:
    with pytest.raises(GemmaInputError) as error:
        adapter_for(RecordingRuntime()).generate(gemma_request)

    assert error.value.code == code
    assert error.value.retryable is False


@pytest.mark.parametrize(
    "response",
    [
        "",
        "not JSON",
        '{"amount":-2,"currency":"INR"}',
    ],
)
def test_invalid_model_responses_are_normalized(response: str) -> None:
    runtime = RecordingRuntime(response)
    adapter = adapter_for(runtime)

    with pytest.raises(GemmaResponseError) as error:
        adapter.generate(
            GemmaRequest(
                prompt="Extract the amount.",
                response_schema=ExtractedAmount,
            )
        )

    assert error.value.code in {
        "gemma_empty_response",
        "gemma_invalid_structured_response",
    }
    assert error.value.retryable is True


def test_runtime_load_and_generation_failures_are_normalized() -> None:
    def broken_factory(_model_id: str) -> RecordingRuntime:
        raise OSError("private model cache path")

    with pytest.raises(GemmaLoadError) as load_error:
        GemmaAdapter(runtime_factory=broken_factory).generate(
            GemmaRequest(prompt="hello")
        )
    assert load_error.value.code == "gemma_model_load_failed"
    assert "private model cache path" not in str(load_error.value)

    runtime = RecordingRuntime()
    runtime.error = RuntimeError("GPU internals")
    with pytest.raises(GemmaInferenceError) as inference_error:
        adapter_for(runtime).generate(GemmaRequest(prompt="hello"))
    assert inference_error.value.code == "gemma_inference_failed"
    assert "GPU internals" not in str(inference_error.value)


def test_runtime_is_loaded_once_and_reused() -> None:
    runtime = RecordingRuntime()
    factory_calls: list[str] = []

    def factory(model_id: str) -> RecordingRuntime:
        factory_calls.append(model_id)
        return runtime

    adapter = GemmaAdapter(runtime_factory=factory)
    adapter.generate(GemmaRequest(prompt="first"))
    adapter.generate(GemmaRequest(prompt="second"))

    assert factory_calls == [DEFAULT_MODEL_ID]
    assert len(runtime.calls) == 2
