import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field, SecretStr

from app.core.config import settings
from app.inference.gemma import (
    DEFAULT_MODEL_ID,
    GemmaAdapter,
    GemmaInferenceError,
    GemmaInputError,
    GemmaLoadError,
    GemmaRequest,
    GemmaResponseError,
    GemmaRuntimeOutput,
    GemmaToolCall,
    TransformersGemma4Runtime,
)


class RecordingRuntime:
    def __init__(self, response: str = "A clear answer") -> None:
        self.output = GemmaRuntimeOutput(text=response)
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stream_chunks: list[str] = ["A ", "clear ", "answer"]
        self.stream_calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: tuple[dict[str, Any], ...],
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> GemmaRuntimeOutput:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "enable_thinking": enable_thinking,
            }
        )
        if self.error:
            raise self.error
        return self.output

    def generate_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ):
        self.stream_calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        if self.error:
            raise self.error
        yield from self.stream_chunks


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
    assert response.model_id == DEFAULT_MODEL_ID == "google/gemma-4-E4B-it"
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
    assert content[3]["type"] == "text"
    assert content[4] == {"type": "audio", "audio": audio_object}


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


def test_native_tool_schemas_and_thinking_are_forwarded() -> None:
    runtime = RecordingRuntime("")
    runtime.output = GemmaRuntimeOutput(
        thinking="Use the calculator.",
        tool_calls=(
            GemmaToolCall(name="add_amounts", arguments={"amounts": [1, 2]}),
        ),
    )
    adapter = adapter_for(runtime)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "add_amounts",
            "description": "Add values.",
            "parameters": {
                "type": "object",
                "properties": {"amounts": {"type": "array"}},
                "required": ["amounts"],
            },
        },
    }

    response = adapter.generate(
        GemmaRequest(
            prompt="Add 1 and 2.",
            tools=(tool_schema,),
            enable_thinking=True,
        )
    )

    assert response.text == ""
    assert response.thinking == "Use the calculator."
    assert response.tool_calls[0].name == "add_amounts"
    assert runtime.calls[0]["tools"] == (tool_schema,)
    assert runtime.calls[0]["enable_thinking"] is True


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
            GemmaRequest(
                prompt="hello",
                response_schema=ExtractedAmount,
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "test",
                            "parameters": {"type": "object"},
                        },
                    },
                ),
            ),
            "conflicting_response_modes",
        ),
        (
            GemmaRequest(prompt="hello", tools=({"type": "invalid"},)),
            "invalid_tool_schema",
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


def test_generate_stream_yields_chunks_from_the_runtime() -> None:
    runtime = RecordingRuntime()
    adapter = adapter_for(runtime)

    chunks = list(
        adapter.generate_stream(
            GemmaRequest(prompt="Explain this notice.", max_new_tokens=64, temperature=0.2)
        )
    )

    assert chunks == ["A ", "clear ", "answer"]
    assert runtime.stream_calls[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Explain this notice."}],
        }
    ]
    assert runtime.stream_calls[0]["max_new_tokens"] == 64
    assert runtime.stream_calls[0]["temperature"] == 0.2


@pytest.mark.parametrize(
    ("gemma_request", "code"),
    [
        (
            GemmaRequest(prompt="hello", response_schema=ExtractedAmount),
            "streaming_unsupported_with_structured_output",
        ),
        (
            GemmaRequest(
                prompt="hello",
                tools=(
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": {"type": "object"}},
                    },
                ),
            ),
            "streaming_unsupported_with_structured_output",
        ),
        (
            GemmaRequest(prompt="hello", enable_thinking=True),
            "streaming_unsupported_with_thinking",
        ),
    ],
)
def test_generate_stream_rejects_structured_or_thinking_requests(
    gemma_request: GemmaRequest,
    code: str,
) -> None:
    adapter = adapter_for(RecordingRuntime())

    with pytest.raises(GemmaInputError) as error:
        adapter.generate_stream(gemma_request)

    assert error.value.code == code


def test_transformers_runtime_uses_gemma4_multimodal_api_and_parse_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "hf_token", SecretStr("hf_test_token_for_unit_tests"))
    calls: dict[str, Any] = {}

    class FakeInputs(dict):
        def to(self, device: str) -> "FakeInputs":
            calls["inputs_device"] = device
            return self

    class FakeInputIds:
        shape = (1, 3)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeProcessor":
            calls["processor_model_id"] = model_id
            calls["processor_options"] = kwargs
            return cls()

        def apply_chat_template(self, messages, **kwargs) -> FakeInputs:
            calls["messages"] = messages
            calls["template_options"] = kwargs
            return FakeInputs(input_ids=FakeInputIds())

        def decode(self, tokens, *, skip_special_tokens: bool) -> str:
            calls["decoded_tokens"] = tokens
            calls["skip_special_tokens"] = skip_special_tokens
            return "raw gemma response"

        def parse_response(self, response: str) -> dict[str, Any]:
            calls["parsed_response"] = response
            return {
                "role": "assistant",
                "thinking": "Need arithmetic.",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add_amounts",
                            "arguments": {"amounts": [1, 2]},
                        },
                    }
                ],
            }

    class FakeModel:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs) -> "FakeModel":
            calls["model_id"] = model_id
            calls["model_options"] = kwargs
            return cls()

        def eval(self) -> "FakeModel":
            return self

        def generate(self, **kwargs):
            calls["generation_options"] = kwargs
            return [[10, 11, 12, 13, 14]]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForMultimodalLM=FakeModel,
        ),
    )
    runtime = TransformersGemma4Runtime(DEFAULT_MODEL_ID)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "add_amounts",
            "parameters": {"type": "object"},
        },
    }

    output = runtime.generate(
        messages=[{"role": "user", "content": "Add 1 and 2."}],
        tools=(tool_schema,),
        max_new_tokens=64,
        temperature=1.0,
        enable_thinking=True,
    )

    assert calls["model_id"] == "google/gemma-4-E4B-it"
    assert calls["processor_options"] == {"token": "hf_test_token_for_unit_tests"}
    assert calls["model_options"] == {
        "device_map": "auto",
        "dtype": "auto",
        "token": "hf_test_token_for_unit_tests",
    }
    assert calls["template_options"]["tools"] == [tool_schema]
    assert calls["template_options"]["enable_thinking"] is True
    assert calls["skip_special_tokens"] is False
    assert calls["parsed_response"] == "raw gemma response"
    assert output.thinking == "Need arithmetic."
    assert output.tool_calls == (
        GemmaToolCall(name="add_amounts", arguments={"amounts": [1, 2]}),
    )
    assert calls["generation_options"]["temperature"] == 1.0
    assert calls["generation_options"]["top_p"] == 0.95
    assert calls["generation_options"]["top_k"] == 64


def test_transformers_runtime_streams_tokens_via_text_iterator_streamer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "hf_token", SecretStr("hf_test_token_for_unit_tests"))
    calls: dict[str, Any] = {}

    class FakeInputs(dict):
        def to(self, device: str) -> "FakeInputs":
            return self

    class FakeProcessor:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeProcessor":
            return cls()

        def apply_chat_template(self, messages, **kwargs) -> FakeInputs:
            calls["messages"] = messages
            calls["template_options"] = kwargs
            return FakeInputs(input_ids=object())

    class FakeModel:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs) -> "FakeModel":
            return cls()

        def eval(self) -> "FakeModel":
            return self

        def generate(self, **kwargs):
            calls["generation_options"] = kwargs

    class FakeStreamer:
        def __init__(self, tokenizer: Any, *, skip_prompt: bool, skip_special_tokens: bool) -> None:
            calls["streamer_tokenizer"] = tokenizer
            calls["streamer_options"] = {
                "skip_prompt": skip_prompt,
                "skip_special_tokens": skip_special_tokens,
            }

        def __iter__(self):
            return iter(["Pay ", "before ", "Friday."])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForMultimodalLM=FakeModel,
            TextIteratorStreamer=FakeStreamer,
        ),
    )
    runtime = TransformersGemma4Runtime(DEFAULT_MODEL_ID)

    chunks = list(
        runtime.generate_stream(
            messages=[{"role": "user", "content": "Explain this notice."}],
            max_new_tokens=64,
            temperature=0.0,
        )
    )

    assert chunks == ["Pay ", "before ", "Friday."]
    assert calls["streamer_tokenizer"] is FakeProcessor.tokenizer
    assert calls["streamer_options"] == {
        "skip_prompt": True,
        "skip_special_tokens": True,
    }
    assert calls["generation_options"]["max_new_tokens"] == 64
    assert calls["generation_options"]["do_sample"] is False
    assert "streamer" in calls["generation_options"]
