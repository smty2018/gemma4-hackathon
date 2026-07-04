# Project Overview

India's multilingual AI citizen agent, powered by Gemma 4.

This project turns difficult public-service documents into evidence-grounded explanations and safe, user-approved action plans. The initial demo focuses on electricity bills and government notices in English, Hindi, and Bengali.

## MVP

- Upload an image or PDF of a supported document.
- Extract amounts, dates, deadlines, penalties, and required actions.
- Explain the document in the user's language and preferred communication style.
- Answer follow-up questions using evidence from the document.
- Propose tools such as calculator, reminder, official search, and office lookup.
- Require confirmation before any tool creates an external side effect.
- Produce text responses and use a separate TTS adapter for spoken output.

## Repository layout

```text
apps/web/                  Mobile-first Next.js interface
services/api/              FastAPI agent and document pipeline
packages/contracts/        Shared JSON contracts
docs/                      Architecture, safety, and demo guidance
```

## Architecture principle

```text
document -> evidence extraction -> typed facts -> action plan
         -> user confirmation -> tool execution -> grounded response
```

The model may propose a tool call, but application code validates its name, arguments, permissions, and confirmation state before execution.

## Local setup

Copy `.env.example` to `.env`, then start each workspace independently.

### API

```powershell
cd services/api
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

### Web

```powershell
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000`. The API health endpoint is `http://localhost:8000/api/v1/health`.

## Status

This first commit establishes the system boundaries and runnable application shells. Gemma inference, PDF rendering, OCR evaluation, and external tool providers are the next implementation milestones.
