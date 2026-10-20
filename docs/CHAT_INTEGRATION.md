# Grounded chat orchestration (prototype only)

The `POST /chat` route supports a controlled, single-user prototype using the
configured Bedrock Knowledge Base and Groq provider. This is **not authentication
or persistent document ownership**. Use it only from localhost or a private
network; do not expose it publicly or through a public proxy.

## Current behavior

- Requests use the strict `ChatRequest` contract in `backend/contracts.py`.
- Before retrieval, the mandatory demo `SourceAuthorizer` validates each
  requested PDF/PPTX ID against the server's upload directory and verifies that
  the corresponding exact object key exists in the configured S3 bucket.
  Client-provided IDs are selectors only; they are not proof of permission.
  The session ID is only a request label, not an identity.
- The Bedrock adapter supplies document evidence using `RetrievalHit` objects.
- Video retrieval is not enabled in this prototype.
- The server checks source IDs and evidence types. It excludes full transcript
  segments if their end time is after the current playback position, even if
  the video retriever supplies them accidentally.
- On no eligible evidence, the service abstains and does not call the model.
- Provider failures become HTTP 503, denied sources become HTTP 403.
- The response lists citation metadata from accepted evidence (not generated
  by the model). These are **candidate supporting citations**, not proven
  entailment of each sentence of the answer.

## Prototype configuration

No per-document environment variable is needed. The application automatically
recognizes upload-generated PDF/PPTX IDs that are present in its local upload
directory and in the configured S3 bucket. Startup requires the existing
`S3_BUCKET_NAME`, `AWS_REGION`, Bedrock retrieval configuration and
`GROQ_API_KEY`; `GROQ_MODEL_NAME` is optional.

This is a bounded prototype policy, not authentication or document ownership.
It lets any caller who can reach the private demo service request any
upload-generated PDF/PPTX currently present in that service's upload directory
and configured bucket. Use one private, single-user instance and a bucket
dedicated to these demo uploads. Keep the service on localhost/private ingress
and do not rely on forwarded proxy headers as a security boundary. A real
deployment requires authenticated identity and persistent session/document
ownership checks before enabling chat.

## Accepted content policy

Documents are treated as separately permitted learning materials. If the
lesson's PDFs or PPTs contain future topics, an additional time/coverage filter
is needed before those materials can be considered safe at a prior playback
timestamp. Video evidence crossing the cutoff is **excluded**, not truncated.
The retrieval team may provide timestamp-split chunks for better coverage.

## Not completed

- Authentication and persistent document/session ownership
- Video retrieval
- Multi-turn chat history or session storage
- Model-account connectivity/deployment smoke test
- Sentence-level citation entailment and hallucination evaluation
- Quiz API and frontend chat UX

The returned citations identify accepted retrieved chunks; they do not prove
sentence-level entailment. Do not publicly enable the route until real
authentication and source ownership controls are implemented.

## Tests

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests/test_chat_service.py -q
```
