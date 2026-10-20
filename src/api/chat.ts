const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export type ChatCitation = {
  source_id: string
  source_type: string
  chunk_id: string
  page: number | null
  slide: number | null
  start_seconds: number | null
  end_seconds: number | null
}

export type ChatRequest = {
  session_id: string
  question: string
  allowed_source_ids: string[]
  video_source_id: string | null
  playback_seconds: number | null
}

export type ChatResponse = {
  session_id: string
  answer: string
  grounded: boolean
  citations: ChatCitation[]
}

export class ChatApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ChatApiError'
    this.status = status
  }
}

function messageForStatus(status: number) {
  if (status === 422) return 'The question could not be understood by the chat service.'
  if (status === 403) return 'This material is not available for the current chat session.'
  if (status === 404) return 'The chat endpoint or selected material was not found.'
  if (status === 503) return 'Grounded chat is not ready yet. The backend integration is still being connected.'
  return 'The chat service returned an unexpected error.'
}

export async function askChat(request: ChatRequest): Promise<ChatResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  } catch {
    throw new ChatApiError('The backend could not be reached. Check that it is running on port 8000.')
  }

  let payload: Partial<ChatResponse> & { detail?: string } = {}
  try {
    payload = await response.json()
  } catch {
    // Use the status-based message when the server did not return JSON.
  }

  if (!response.ok) {
    throw new ChatApiError(payload.detail || messageForStatus(response.status), response.status)
  }
  if (
    typeof payload.session_id !== 'string' ||
    typeof payload.answer !== 'string' ||
    typeof payload.grounded !== 'boolean' ||
    !Array.isArray(payload.citations)
  ) {
    throw new ChatApiError('The chat service returned an incomplete response.', response.status)
  }

  return payload as ChatResponse
}
