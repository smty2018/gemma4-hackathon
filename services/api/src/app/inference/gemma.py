import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

DEFAULT_MODEL_ID = "google/gemma-3n-E4B-it"
MAX_GENERATED_TOKENS = 4_096


class GemmaAdapterError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class GemmaInputError(GemmaAdapterError):
    pass


class GemmaDependencyError(GemmaAdapterError):
    pass


class GemmaLoadError(GemmaAdapterError):
    pass


class GemmaInferenceError(GemmaAdapterError):
    pass


class GemmaResponseError(GemmaAdapterError):
    pass


StructuredSchema = type[BaseModel] | dict[str, Any]


@dataclass(frozen=True)
class GemmaRequest:
    prompt: str = ""
    images: Sequence[Any] = field(default_factory=tuple)
    audio: Sequence[Any] = field(default_factory=tuple)
    response_schema: StructuredSchema | None = None
    system_instruction: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    model_id: str
    structured: dict[str, Any] | None = None


class GemmaRuntime(Protocol):
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...


class TransformersGemmaRuntime:
    def __init__(self, model_id: str) -> None:
        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        except ImportError as error:
            raise GemmaDependencyError(
                "gemma_dependencies_missing",
                "Gemma runtime dependencies are not installed.",
            ) from error

        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
            ).eval()
        except Exception as error:
            raise GemmaLoadError(
                "gemma_model_load_failed",
                "Gemma could not be loaded. Check model access, storage, and memory.",
                retryable=True,
            ) from error

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generation_options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature

        generated = self._model.generate(**inputs, **generation_options)
        prompt_length = inputs["input_ids"].shape[-1]
        return self._processor.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()


RuntimeFactory = Callable[[str], GemmaRuntime]


class GemmaAdapter:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        runtime_factory: RuntimeFactory = TransformersGemmaRuntime,
    ) -> None:
        if not model_id.strip():
            raise GemmaInputError(
                "invalid_model_id",
                "A Gemma model ID is required.",
            )
        self.model_id = model_id
        self._runtime_factory = runtime_factory
        self._runtime: GemmaRuntime | None = None
        self._runtime_lock = Lock()

    def generate(self, request: GemmaRequest) -> GemmaResponse:
        schema, response_model = self._resolve_schema(request.response_schema)
        self._validate_request(request)
        messages = self._build_messages(request, schema)

        try:
            raw_text = self._get_runtime().generate(
                messages=messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
        except GemmaAdapterError:
            raise
        except Exception as error:
            raise GemmaInferenceError(
                "gemma_inference_failed",
                "Gemma could not complete this request.",
                retryable=True,
            ) from error

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise GemmaResponseError(
                "gemma_empty_response",
                "Gemma returned an empty response.",
                retryable=True,
            )

        text = raw_text.strip()
        if schema is None:
            return GemmaResponse(text=text, model_id=self.model_id)

        structured = self._parse_structured_response(
            text,
            schema=schema,
            response_model=response_model,
        )
        return GemmaResponse(
            text=text,
            model_id=self.model_id,
            structured=structured,
        )

    def _get_runtime(self) -> GemmaRuntime:
        if self._runtime is not None:
            return self._runtime

        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                self._runtime = self._runtime_factory(self.model_id)
            except GemmaAdapterError:
                raise
            except Exception as error:
                raise GemmaLoadError(
                    "gemma_model_load_failed",
                    "Gemma could not be loaded. Check model access, storage, and memory.",
                    retryable=True,
                ) from error
            return self._runtime

    @staticmethod
    def _validate_request(request: GemmaRequest) -> None:
        if not request.prompt.strip() and not request.images and not request.audio:
            raise GemmaInputError(
                "empty_gemma_request",
                "Provide text, an image, or audio for Gemma to process.",
            )
        if any(source is None for source in (*request.images, *request.audio)):
            raise GemmaInputError(
                "invalid_media_source",
                "Image and audio sources cannot be empty.",
            )
        if len(request.audio) > 1:
            raise GemmaInputError(
                "too_many_audio_inputs",
                "Gemma 3n accepts one target audio clip per request.",
            )
        if not 1 <= request.max_new_tokens <= MAX_GENERATED_TOKENS:
            raise GemmaInputError(
                "invalid_generation_limit",
                f"max_new_tokens must be between 1 and {MAX_GENERATED_TOKENS}.",
            )
        if not 0 <= request.temperature <= 2:
            raise GemmaInputError(
                "invalid_temperature",
                "temperature must be between 0 and 2.",
            )

    @classmethod
    def _build_messages(
        cls,
        request: GemmaRequest,
        schema: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system_instruction and request.system_instruction.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": request.system_instruction.strip(),
                        }
                    ],
                }
            )

        content: list[dict[str, Any]] = []
        content.extend(cls._media_part("image", source) for source in request.images)
        content.extend(cls._media_part("audio", source) for source in request.audio)

        instructions: list[str] = []
        if request.prompt.strip():
            instructions.append(request.prompt.strip())
        if schema is not None:
            instructions.append(
                "Return only one valid JSON object with no Markdown fencing or commentary. "
                "The object must satisfy this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        if instructions:
            content.append({"type": "text", "text": "\n\n".join(instructions)})

        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _media_part(media_type: str, source: Any) -> dict[str, Any]:
        if isinstance(source, Path):
            return {"type": media_type, "path": str(source)}
        if isinstance(source, str):
            key = "url" if source.startswith(("http://", "https://", "data:")) else "path"
            return {"type": media_type, key: source}
        return {"type": media_type, media_type: source}

    @staticmethod
    def _resolve_schema(
        response_schema: StructuredSchema | None,
    ) -> tuple[dict[str, Any] | None, type[BaseModel] | None]:
        if response_schema is None:
            return None, None

        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            schema = response_schema.model_json_schema()
            return schema, response_schema

        if not isinstance(response_schema, dict):
            raise GemmaInputError(
                "invalid_response_schema",
                "response_schema must be a JSON Schema object or Pydantic model.",
            )

        try:
            Draft202012Validator.check_schema(response_schema)
        except SchemaError as error:
            raise GemmaInputError(
                "invalid_response_schema",
                "The supplied JSON Schema is invalid.",
            ) from error
        return response_schema, None

    @staticmethod
    def _parse_structured_response(
        text: str,
        *,
        schema: dict[str, Any],
        response_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        try:
            parsed = _decode_json_object(text)
            if response_model is not None:
                return response_model.model_validate(parsed).model_dump(mode="json")
            Draft202012Validator(schema).validate(parsed)
            return parsed
        except (JSONDecodeError, JsonSchemaValidationError, PydanticValidationError) as error:
            raise GemmaResponseError(
                "gemma_invalid_structured_response",
                "Gemma returned JSON that did not match the requested structure.",
                retryable=True,
            ) from error


def _decode_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()

    decoder = JSONDecoder()
    try:
        parsed = json.loads(candidate)
    except JSONDecodeError:
        parsed = None
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate, index)
                break
            except JSONDecodeError:
                continue
        if parsed is None:
            raise

    if not isinstance(parsed, dict):
        raise JSONDecodeError("Expected a JSON object", candidate, 0)
    return parsed
