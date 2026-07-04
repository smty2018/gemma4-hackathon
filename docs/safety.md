# Safety contract

The citizen agent handles documents that can affect money, benefits, health, identity, and legal rights. The application therefore follows these rules:

- Never claim that an eligibility estimate is an official determination.
- Prefer official government sources and display the source URL and retrieval date.
- Distinguish document evidence from model inference.
- Mark low-confidence text and ask the user to verify it against the original page.
- Never submit forms, make payments, share identifiers, or create reminders without confirmation.
- Show the exact tool action and arguments before execution.
- Keep a narrow allow-list of tools; never execute model-generated code.
- Provide escalation guidance for legal, medical, financial, or emergency questions.
- Minimize retention and give users a clear delete control.

The live demo should visibly show these safeguards rather than relegating them to a disclaimer.
