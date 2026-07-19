# AI HealthHack backend

FastAPI backend with two separate capabilities:

- `POST /api/v1/ndis-navigation/plan` creates a structured NDIS navigation plan
  from clinical and participant context using a Hugging Face inference provider.
- `POST /api/v1/patient-chat/message` provides a patient-facing NDIS conversation
  that asks follow-up questions and suggests next actions.
- `WS /ws/transcribe` provides low-latency Whisper transcription for local,
  persistent-service deployments.

## Project layout

```text
app/
  api/routers/       # HTTP and WebSocket endpoints
  core/              # configuration
  schemas/           # validated API contracts
  services/          # NDIS LLM and Whisper integrations
api/index.py         # Vercel ASGI handler
main.py              # local development entry point
```

## Local setup

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The server listens on `PORT` (default `8080`). Copy only the environment
variable names from `.env.example`; never commit a provider access token.

## NDIS navigation API

`POST /api/v1/ndis-navigation/plan`

```json
{
  "clinical_extraction": {
    "diagnosis_reason": "Right MCA stroke with left hemiparesis",
    "mobility_status": "Requires one-person assistance for transfers"
  },
  "ndis_context": {
    "has_active_plan": true,
    "management_type": "Plan-managed",
    "urgency_level": "High"
  }
}
```

The response is validated before it is returned and always contains:

- `practical_needs_summary`
- `recommended_support_categories`
- `provider_referral_summary`
- `call_script`
- `next_steps_checklist`

Set `HUGGINGFACEHUB_API_TOKEN` and `HUGGINGFACE_MODEL`. A value such as
`google/gemma-4-31B-it:novita` is interpreted as model
`google/gemma-4-31B-it` using the `novita` Hugging Face inference provider.

## Discharge document upload

`POST /api/v1/ndis-navigation/document-plan` accepts a discharge-summary
document, extracts explicit clinical and functional information with LangChain,
then produces the same validated NDIS navigation plan. It accepts text-based
PDFs plus PNG and JPEG photos/scans, up to `DOCUMENT_MAX_UPLOAD_BYTES`
(10 MB by default).

```bash
curl -X POST "http://localhost:8080/api/v1/ndis-navigation/document-plan" \
  -F "document=@discharge-summary.pdf;type=application/pdf" \
  -F 'ndis_context={"has_active_plan":false,"urgency_level":"High"}'
```

The response includes `extracted_clinical_information`, the complete `plan`,
and a short `source_text_preview`. Uploaded files are processed in memory and
are not stored by this backend; however, their extracted text or image content
is sent to the configured Hugging Face inference provider. Do not upload real
patient information unless your privacy, consent, and provider agreements
permit this. Scanned PDFs with no selectable text should be uploaded as clear
PNG or JPEG page images.

## Patient NDIS chatbot API

## Discharge action pack

`POST /api/v1/ndis-navigation/action-pack` accepts the same multipart
`document` and optional `ndis_context` fields as the document-plan endpoint.
It returns evidence requirements, access or review guidance, provider service
categories, referral content, family call/email scripts, and prioritised tasks.
It does not search or guarantee local provider availability.

`POST /api/v1/patient-chat/message`

The API is stateless: the frontend owns the conversation and sends the prior
turns needed for the next response. The backend does not persist the messages.
Send only the minimum necessary health and personal information because the
request is processed by the configured third-party AI inference provider.

```json
{
  "message": "Since leaving hospital I cannot shower safely without help.",
  "history": [
    {
      "role": "user",
      "content": "I need help understanding what support I can ask for."
    },
    {
      "role": "assistant",
      "content": "I can help identify practical support needs. What has changed?"
    }
  ],
  "ndis_context": {
    "has_active_plan": true,
    "management_type": "Plan-managed"
  }
}
```

The response provides a plain-language `reply`, up to two
`follow_up_questions`, NDIS-aligned `recommendations` when enough information
is available, and `urgent_action` / `urgent_message` for immediate safety
concerns. It includes a mandatory disclaimer and is not medical, legal, or
emergency advice. The client cannot submit system messages; history must
alternate user and assistant turns and end with an assistant turn.

Limits are controlled by `CHAT_HISTORY_LIMIT`,
`CHAT_MESSAGE_MAX_CHARACTERS`, and `PATIENT_CHAT_TIMEOUT_SECONDS`. Keep
`ALLOWED_ORIGINS` restricted to your production frontend before deployment.

## Transcription WebSocket

Connect to `ws://localhost:8080/ws/transcribe`, then send:

```json
{"type": "audio_chunk", "data": "<base64-int16-pcm>"}
```

Finish with `{"type": "stop"}`. The service sends `partial` and `final` JSON
messages. Whisper models load lazily on the first WebSocket connection.

## Testing

```bash
pytest
```

## Deployment

The NDIS HTTP endpoint is suitable for a serverless ASGI deployment. The
transcription endpoint requires a persistent, WebSocket-capable runtime;
Vercel serverless functions do not support this workload. Deploy that service
to a WebSocket-capable host or run it separately.
