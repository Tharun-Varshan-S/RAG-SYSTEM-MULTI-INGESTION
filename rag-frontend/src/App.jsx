import { useState, useRef, useEffect } from 'react'
import { Bot, User, Send, Sparkles, FileText, Database, Layers, Code2, Video } from 'lucide-react'
import './App.css'

const modeOptions = [
  { id: 'eli5', label: "Explain like I'm 5" },
  { id: 'technical', label: 'Technical' },
  { id: 'example', label: 'Real-world example' },
  { id: 'assist', label: 'Assist' },
]

const initialSources = [
  { id: 1, name: 'BARACK OBAMA', type: 'code' },
  { id: 2, name: 'CODE CHECK', type: 'code' },
  { id: 3, name: 'LLM 2', type: 'youtube' },
]

function App() {
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedMode, setSelectedMode] = useState('assist')
  const [sourceList, setSourceList] = useState(initialSources)
  const [sourceName, setSourceName] = useState('')
  const [sourceType, setSourceType] = useState('document')
  const [sourceText, setSourceText] = useState('')
  const [notification, setNotification] = useState('')
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatHistory, isLoading])

  const buildRequestBody = (query) => {
    const request = {
      question: query,
      assistant_mode: selectedMode === 'eli5' ? 'teach' : 'assist',
      mode: selectedMode === 'technical' ? 'expert' : selectedMode === 'eli5' ? 'beginner' : 'normal',
    }

    if (selectedMode === 'eli5') {
      request.question = `${query} Explain this like I'm 5.`
    }
    if (selectedMode === 'example') {
      request.question = `${query} Please include a real-world example.`
    }

    return request
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
    if (!sourceText.trim() || !sourceName.trim()) {
      setNotification('Provide both a source name and content before ingesting.')
      return
    }

    try {
      const response = await fetch('http://localhost:8000/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: sourceText,
          source_name: sourceName,
          source_type: sourceType,
        }),
      })
      if (!response.ok) {
        throw new Error('Failed to ingest source. Check the backend and try again.')
      }
      const result = await response.json()
      setSourceList(prev => [
        { id: Date.now(), name: sourceName, type: sourceType },
        ...prev,
      ])
      setSourceName('')
      setSourceText('')
      setNotification(`Ingested ${result.chunks_added} chunks from ${sourceName}.`)
    } catch (error) {
      setNotification(error.message)
    }
  }

  const renderSourceIcon = (type) => {
    if (type === 'video' || type === 'youtube') return <Video size={16} />
    if (type === 'code') return <Code2 size={16} />
    return <FileText size={16} />
  }

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
            <p>Upload text, code notes, or video transcripts.</p>
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
                <option value="video">Video</option>
                <option value="pdf">PDF</option>
              </select>
            </label>
            <label>
              Source content
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                placeholder="Paste text, transcript, notes, or code here"
              />
            </label>
            <button type="submit" className="primary-button">Ingest source</button>
            {notification && <div className="form-error">{notification}</div>}
          </form>
        </div>
      </div>

      <div className="right-panel">
        <div className="chat-panel-header">
          <h2>Chat with OmniBrain</h2>
          <p>Ask your knowledge base questions and get grounded answers fast.</p>
        </div>

        <div className="mode-controls">
          {modeOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              className={`mode-button ${selectedMode === option.id ? 'active' : ''}`}
              onClick={() => setSelectedMode(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>

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
          {isLoading && (
            <div className="message-wrapper bot">
              <div className="avatar bot"><Bot size={20} /></div>
              <div className="message-content typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
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
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
