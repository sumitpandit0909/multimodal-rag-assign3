import React, { useState, useEffect, useRef } from 'react';
import { ArrowUp, Sparkles, RefreshCw, UploadCloud } from 'lucide-react';
import { MessageBubble } from './components/MessageBubble';
import { VisualModal } from './components/VisualModal';
import { ExcelDataCard } from './components/ExcelDataCard';
import { IngestionModal } from './components/IngestionModal';
import type { Message, SourceNode, ChatResponse } from './types';
import './App.css';

const CHAT_API_URL = 'http://localhost:8000';
const INGESTION_API_URL = 'http://localhost:8001';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sessionId] = useState<string>(() => 'session_' + Math.random().toString(36).substring(2, 9));
  const [healthStatus, setHealthStatus] = useState<string>('Checking...');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(false);

  // Modals state
  const [selectedVisualSource, setSelectedVisualSource] = useState<SourceNode | null>(null);
  const [selectedExcelSource, setSelectedExcelSource] = useState<SourceNode | null>(null);
  const [isIngestionOpen, setIsIngestionOpen] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Check health endpoint
  const checkHealth = async () => {
    try {
      const res = await fetch(`${CHAT_API_URL}/health`);
      const data = await res.json();
      if (data.status === 'healthy') {
        setIsBackendHealthy(true);
        setHealthStatus(
          data.vectors_ingested_count > 0
            ? `${data.vectors_ingested_count} vectors ready`
            : 'Atlas Connected (0 vectors)'
        );
      } else {
        setIsBackendHealthy(false);
        setHealthStatus('Database Connecting...');
      }
    } catch {
      setIsBackendHealthy(false);
      setHealthStatus('Backend Offline');
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleCitationClick = (source: SourceNode) => {
    if (source.source_type === 'visual') {
      setSelectedVisualSource(source);
    } else {
      setSelectedExcelSource(source);
    }
  };

  const handleSendMessage = async (e?: React.FormEvent, customPrompt?: string) => {
    if (e) e.preventDefault();
    const query = (customPrompt || inputQuery).trim();
    if (!query || isLoading) return;

    setInputQuery('');

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch(`${CHAT_API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: query,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: data.answer,
        sources: data.sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err: any) {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          text: `⚠️ Could not reach chat service (${err.message}). Ensure backend is active on ${CHAT_API_URL}.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-branding">
          <div className="logo-badge">
            <Sparkles size={18} />
          </div>
          <div>
            <h1>Enterprise Knowledge Assistant</h1>
            <p className="subtext">Decoupled Multimodal RAG with Visual Verification</p>
          </div>
        </div>

        <div className="header-actions">
          <button 
            onClick={() => setIsIngestionOpen(true)}
            className="ingest-nav-btn"
            title="Ingest Documents & View Library"
          >
            <UploadCloud size={14} />
            <span>Ingest & Library</span>
          </button>

          <div className="header-status">
            <span className={`status-dot ${isBackendHealthy ? 'dot-green' : 'dot-amber'}`} />
            <span className="status-text">{healthStatus}</span>
            <button onClick={checkHealth} title="Refresh connection" className="refresh-btn">
              <RefreshCw size={12} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Chat Stream */}
      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="empty-hero">
            <div className="hero-icon-ring">
              <Sparkles size={28} />
            </div>
            <h2>What would you like to verify today?</h2>
            <p>
              Ask questions across corporate presentations, documents, and spreadsheets.
              Citations link directly to page screenshots and tabular datasets.
            </p>

            <div className="prompt-suggestions">
              <div
                className="suggestion-pill"
                onClick={() => handleSendMessage(undefined, "So i want to prepare a deck for casinos can you find me some suitable content for that")}
              >
                <span className="pill-tag">Visual</span>
                <span>"So i want to prepare a deck for casinos can you find me some suitable content for that"</span>
              </div>
              <div
                className="suggestion-pill"
                onClick={() => handleSendMessage(undefined, "Do we have a content deck for VIX Micro ?")}
              >
                <span className="pill-tag">Deck</span>
                <span>"Do we have a content deck for VIX Micro ?"</span>
              </div>
              <div
                className="suggestion-pill"
                onClick={() => handleSendMessage(undefined, "Is there a slide that i can use for Bilingualism & Biculturalism ?")}
              >
                <span className="pill-tag">Slide</span>
                <span>"Is there a slide that i can use for Bilingualism & Biculturalism ?"</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map(msg => (
              <MessageBubble
                key={msg.id}
                sender={msg.sender}
                text={msg.text}
                sources={msg.sources}
                onCitationClick={handleCitationClick}
              />
            ))}

            {isLoading && (
              <div className="loading-row">
                <div className="pulse-dot" />
                <span>Searching vectors and synthesizing grounded response...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Floating Input Tray */}
      <footer className="input-tray">
        <form onSubmit={handleSendMessage} className="input-box-wrapper">
          <input
            type="text"
            value={inputQuery}
            onChange={e => setInputQuery(e.target.value)}
            placeholder="Ask a question across PDFs, PPTs, or Excel sheets..."
            disabled={isLoading}
            className="chat-input"
          />
          <button
            type="submit"
            disabled={isLoading || !inputQuery.trim()}
            className="send-button"
            title="Send inquiry"
          >
            <ArrowUp size={18} />
          </button>
        </form>
      </footer>

      {/* Visual Verification Modal */}
      <VisualModal
        isOpen={!!selectedVisualSource}
        onClose={() => setSelectedVisualSource(null)}
        source={selectedVisualSource}
      />

      {/* Excel Data Inspection Modal */}
      <ExcelDataCard
        isOpen={!!selectedExcelSource}
        onClose={() => setSelectedExcelSource(null)}
        source={selectedExcelSource}
      />

      {/* Ingestion & Library Modal */}
      <IngestionModal
        isOpen={isIngestionOpen}
        onClose={() => setIsIngestionOpen(false)}
        apiBaseUrl={INGESTION_API_URL}
        onIngestionSuccess={checkHealth}
      />
    </div>
  );
}
