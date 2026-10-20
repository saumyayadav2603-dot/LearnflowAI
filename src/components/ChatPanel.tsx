import { useState } from 'react'
import { askChat, ChatApiError, type ChatCitation } from '../api/chat'
import './ChatPanel.css'

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  text: string
  grounded?: boolean
  citations?: ChatCitation[]
}

type ChatPanelProps = {
  sourceId: string | null
  sourceType: 'pdf' | 'pptx' | null
}

function citationLabel(citation: ChatCitation) {
  if (citation.source_type === 'pdf' && citation.page !== null) return `PDF page ${citation.page}`
  if (citation.source_type === 'pptx' && citation.slide !== null) return `PPTX slide ${citation.slide}`
  if (citation.start_seconds !== null && citation.end_seconds !== null) {
    return `Video ${citation.start_seconds}s–${citation.end_seconds}s`
  }
  return citation.source_type
}

export function ChatPanel({ sourceId, sourceType }: ChatPanelProps) {
  const [question, setQuestion] = useState('')
  const [sessionId] = useState(() => `session-${crypto.randomUUID()}`)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const submitQuestion = async () => {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isLoading) {
      if (!trimmedQuestion) setError('Enter a question before sending it.')
      return
    }
    if (!sourceId || !sourceType) {
      setError('Upload a PDF or PPTX and wait for it to finish uploading before asking a question.')
      return
    }

    setError('')
    setQuestion('')
    setMessages((current) => [...current, { id: Date.now(), role: 'user', text: trimmedQuestion }])
    setIsLoading(true)
    try {
      const response = await askChat({
        session_id: sessionId,
        question: trimmedQuestion,
        allowed_source_ids: [sourceId],
        video_source_id: null,
        playback_seconds: null,
      })
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          text: response.answer,
          grounded: response.grounded,
          citations: response.citations,
        },
      ])
    } catch (caughtError) {
      const message = caughtError instanceof ChatApiError
        ? caughtError.message
        : 'The question could not be sent. Please try again.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <article className="chat-card" aria-labelledby="chat-title">
      <div className="chat-heading">
        <div>
          <p className="card-kicker">GROUNDED AI CHAT</p>
          <h2 id="chat-title">Ask about your material</h2>
        </div>
        <span className="chat-session-label">This session only</span>
      </div>
      <p className="chat-description">
        Ask a question about the uploaded PDF or PPTX. Answers and citations come from the backend.
      </p>

      <div className="chat-messages" aria-live="polite">
        {messages.length === 0 && (
          <p className="chat-empty">Your questions and grounded answers will appear here.</p>
        )}
        {messages.map((message) => (
          <div className={`chat-message ${message.role}`} key={message.id}>
            <span className="chat-role">{message.role === 'user' ? 'You' : 'LearnFlow AI'}</span>
            <p>{message.text}</p>
            {message.role === 'assistant' && message.grounded === false && (
              <span className="unsupported-label">Unsupported by the available learning material</span>
            )}
            {message.citations && message.citations.length > 0 && (
              <div className="citation-list">
                {message.citations.map((citation) => (
                  <span className="citation" key={`${citation.chunk_id}-${citation.source_id}`}>
                    {citationLabel(citation)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && <div className="chat-loading" role="status"><span /> Finding grounded evidence...</div>}
      </div>

      <form className="chat-form" onSubmit={(event) => { event.preventDefault(); void submitQuestion() }}>
        <label className="sr-only" htmlFor="chat-question">Ask a question</label>
        <input
          id="chat-question"
          value={question}
          onChange={(event) => { setQuestion(event.target.value); if (error) setError('') }}
          placeholder="e.g. What is the main idea?"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
      {error && <p className="chat-error" role="alert">{error}</p>}
    </article>
  )
}
