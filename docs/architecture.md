# Architecture

## System boundary

The citizen agent is a decision-support assistant, not an autonomous authority. It may explain evidence and propose actions, but external side effects require explicit user confirmation.

## Processing flow

1. **Ingest** — validate type and size, strip metadata where appropriate, and assign an ephemeral document ID.
2. **Render** — convert PDFs into page images while preserving page numbers.
3. **Extract** — ask Gemma for structured facts with source spans and confidence values.
4. **Verify** — parse dates and currency deterministically; reject unsupported or contradictory facts.
5. **Explain** — generate a multilingual, audience-aware explanation from verified facts.
6. **Plan** — propose zero or more allow-listed tool calls.
7. **Confirm** — show the action, destination, and data that would be shared.
8. **Execute** — validate arguments again and call the selected adapter.
9. **Respond** — summarize the result with document evidence and official links.

## Components

- `apps/web`: accessible, mobile-first upload and conversation interface.
- `services/api`: authentication boundary, upload handling, orchestration, and tool execution.
- Gemma runtime: multimodal extraction, multilingual explanation, and tool proposals.
- deterministic validators: dates, currency, file types, and schema validation.
- tool adapters: calculator, reminder, maps, and official government search.
- TTS adapter: converts final text to speech; Gemma remains the text-generating component.

## Data policy

- Uploaded citizen documents are not committed, logged, or retained by default.
- Extracted facts remain tied to page-level evidence.
- Provider requests must disclose which data leaves the device.
- Sensitive identifiers should be redacted before telemetry or debugging.
