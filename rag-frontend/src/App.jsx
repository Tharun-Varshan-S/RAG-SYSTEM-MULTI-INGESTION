import { useState, useRef, useEffect } from 'react'
import { Bot, User, Send, FileText, Layers, Code2, Video, Loader2 } from 'lucide-react'
import './App.css'

const sourceHelpText = {
  document: 'Paste copied text from a document, article, or note.',
  code: 'Paste code snippets, snippets from docs, or repo notes.',
  pdf: 'Upload a PDF file and OmniBrain will extract the text automatically.',
  youtube: 'Paste a YouTube URL and OmniBrain will fetch the transcript automatically.',
}

const normalizeSourceType = (value) => (value === 'video' ? 'youtube' : value)

function App() {
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sourceList, setSourceList] = useState([])
  const [sourceName, setSourceName] = useState('')
  const [sourceType, setSourceType] = useState('document')
  const [sourceText, setSourceText] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [sourceFile, setSourceFile] = useState(null)
  const [notification, setNotification] = useState('')
  const [isIngesting, setIsIngesting] = useState(false)
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatHistory, isLoading])

  const fetchSources = async () => {
    try {
      const response = await fetch('http://localhost:8000/sources')
      if (!response.ok) return
      const payload = await response.json()
      const sources = Array.isArray(payload?.sources)
        ? payload.sources.map((item, index) => ({
          id: `${item.name}-${item.type}-${index}`,
          name: item.name,
          type: item.type,
        }))
        : []
      setSourceList(sources)
    } catch {
      // Keep the current source list when the backend is unavailable.
    }
  }

  useEffect(() => {
    fetchSources()
  }, [])

  const buildRequestBody = (query) => {
    return {
      question: query,
      assistant_mode: 'assist',
      mode: 'normal',
    }
  }

  const askAPI = async (payload) => {
    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        throw new Error('Unable to reach OmniBrain backend. Please ensure the server is running on port 8000.')
      }
      return await response.json()
    } catch (err) {
      if (err.message.includes('Failed to fetch') || err.message.includes('reach OmniBrain backend')) {
        throw new Error('Unable to connect to OmniBrain. Start the backend at http://localhost:8000.')
      }
      throw err
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return

    const userMessage = { type: 'user', text: question.trim() }
    setChatHistory(prev => [...prev, userMessage])
    setQuestion('')
    setIsLoading(true)
    setNotification('')

    const payload = buildRequestBody(question.trim())

    try {
      const resp = await askAPI(payload)
      const botText = [
        `Understanding: ${resp.understanding}`,
        resp.key_points?.length ? `Key points:\n- ${resp.key_points.join('\n- ')}` : null,
        `Explanation: ${resp.explanation}`,
        resp.real_world_example ? `Real-world example: ${resp.real_world_example}` : null,
        resp.next_steps?.length ? `Next steps:\n- ${resp.next_steps.join('\n- ')}` : null,
      ].filter(Boolean).join('\n\n')

      setChatHistory(prev => [...prev, {
        type: 'bot',
        text: botText,
        sources: resp.sources || [],
      }])
    } catch (error) {
      setChatHistory(prev => [...prev, {
        type: 'bot',
        text: error.message,
        isError: true,
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleIngest = async (e) => {
    e.preventDefault()
    if (isIngesting) return
    const normalizedSourceType = normalizeSourceType(sourceType)

    if (!sourceName.trim()) {
      setNotification('Provide a source name before ingesting.')
      return
    }

    if (normalizedSourceType === 'pdf' && !sourceFile) {
      setNotification('Choose a PDF file before ingesting.')
      return
    }

    if (normalizedSourceType === 'youtube' && !sourceUrl.trim()) {
      setNotification('Paste a YouTube URL before ingesting.')
      return
    }

    if ((normalizedSourceType === 'document' || normalizedSourceType === 'code') && !sourceText.trim()) {
      setNotification('Paste the source text before ingesting.')
      return
    }

    try {
      setIsIngesting(true)
      setNotification('')
      const formData = new FormData()
      formData.append('source_type', normalizedSourceType)
      formData.append('source_name', sourceName.trim())

      if (normalizedSourceType === 'pdf') {
        formData.append('file', sourceFile)
      } else if (normalizedSourceType === 'youtube') {
        formData.append('source_url', sourceUrl.trim())
      } else {
        formData.append('source_text', sourceText.trim())
      }

      const response = await fetch('http://localhost:8000/ingest-source', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null)
        throw new Error(errorPayload?.detail || 'Failed to ingest source. Check the backend and try again.')
      }
      const result = await response.json()
      await fetchSources()
      setSourceName('')
      setSourceText('')
      setSourceUrl('')
      setSourceFile(null)
      setNotification(`Ingested ${result.chunks_added} chunks from ${sourceName}.`)
    } catch (error) {
      setNotification(error.message)
    } finally {
      setIsIngesting(false)
    }
  }

  const renderSourceIcon = (type) => {
    if (type === 'video' || type === 'youtube') return <Video size={16} />
    if (type === 'code') return <Code2 size={16} />
    return <FileText size={16} />
  }

  const activeSourceType = normalizeSourceType(sourceType)

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (question.trim() && !isLoading) handleSubmit(e)
    }
  }

  return (
    <div className="app-shell">
      <div className="left-panel">
        <div className="panel-brand">
          <div className="brand-icon">
            <Layers size={20} />
          </div>
          <div>
            <h2>OmniBrain</h2>
            <p>Universal Knowledge Copilot</p>
          </div>
        </div>

        <div className="panel-section">
          <div className="panel-section-header">
            <h3>Ingested sources</h3>
            <p>Sources available for retrieval.</p>
          </div>
          <div className="panel-list">
            {sourceList.length === 0 ? (
              <div className="panel-empty">No sources ingested yet.</div>
            ) : (
              sourceList.map((item) => (
                <div key={item.id} className="source-item">
                  <div className="source-icon">{renderSourceIcon(item.type)}</div>
                  <div>
                    <div className="source-title">{item.name}</div>
                    <div className="source-meta">{item.type.toUpperCase()}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="panel-section">
          <div className="panel-section-header">
            <h3>Add Knowledge</h3>
            <p>Upload PDFs, paste document or code text, or provide a YouTube URL.</p>
          </div>
          <form className="ingest-form" onSubmit={handleIngest}>
            <label>
              Source name
              <input
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                placeholder="e.g. Project README"
              />
            </label>
            <label>
              Source type
              <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
                <option value="document">Document</option>
                <option value="code">Code</option>
                <option value="pdf">PDF</option>
                <option value="youtube">YouTube</option>
              </select>
            </label>
            <div className="source-help">{sourceHelpText[activeSourceType]}</div>
            {activeSourceType === 'pdf' ? (
              <label>
                PDF file
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => setSourceFile(event.target.files?.[0] || null)}
                />
                {sourceFile && <div className="file-name">Selected file: {sourceFile.name}</div>}
              </label>
            ) : activeSourceType === 'youtube' ? (
              <label>
                YouTube URL
                <input
                  type="url"
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                />
              </label>
            ) : (
              <label>
                Source content
                <textarea
                  value={sourceText}
                  onChange={(e) => setSourceText(e.target.value)}
                  placeholder="Paste text, notes, or code here"
                />
              </label>
            )}
            <button type="submit" className="primary-button" disabled={isIngesting}>
              {isIngesting ? (
                <>
                  <Loader2 size={16} className="spin" />
                  Ingesting...
                </>
              ) : (
                'Ingest source'
              )}
            </button>
            {notification && <div className="form-error">{notification}</div>}
          </form>
        </div>
      </div>

      <div className="right-panel">
        <div className="chat-panel-header">
          <h2>Chat with OmniBrain</h2>
          <p>Ask your knowledge base questions and get direct grounded answers.</p>
        </div>

        {isLoading && (
          <div className="chat-loader-banner">
            <Loader2 size={16} className="spin" />
            OmniBrain is preparing a grounded answer...
          </div>
        )}

        <div className="chat-box">
          {chatHistory.length === 0 ? (
            <div className="panel-empty">
              Enter a question on the right to start your OmniBrain session.
            </div>
          ) : (
            chatHistory.map((msg, index) => (
              <div key={index} className={`message-wrapper ${msg.type} ${msg.isError ? 'error' : ''}`}>
                <div className={`avatar ${msg.type}`}>
                  {msg.type === 'user' ? <User size={20} color="white" /> : <Bot size={20} />}
                </div>
                <div className="message-content">
                  {msg.type === 'bot' ? (
                    <div>
                      {msg.text.split('\n\n').map((block, idx) => (
                        <div key={idx} className="response-block">
                          {block.split('\n').map((line, li) => (
                            <p key={li}>{line}</p>
                          ))}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p>{msg.text}</p>
                  )}
                  {msg.sources?.length > 0 && (
                    <div className="sources-container">
                      {msg.sources.map((src, j) => (
                        <span key={j} className="source-badge">
                          <FileText size={12} />
                          {src?.source_name || 'source'} ({src?.source || 'type'})
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-footer">
          <form className="input-box" onSubmit={handleSubmit}>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask OmniBrain to summarize a video, explain a concept, or debug code..."
              disabled={isLoading}
              rows={2}
            />
            <button type="submit" className="send-btn" disabled={isLoading || !question.trim()}>
              {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
