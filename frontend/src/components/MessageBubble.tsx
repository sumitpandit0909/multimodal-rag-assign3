import React from 'react';
import { Eye, Table as TableIcon } from 'lucide-react';
import type { SourceNode } from '../types';

interface MessageBubbleProps {
  sender: 'user' | 'assistant';
  text: string;
  sources?: SourceNode[];
  onCitationClick: (source: SourceNode) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  sender,
  text,
  sources = [],
  onCitationClick,
}) => {
  const isUser = sender === 'user';

  // Parse [1], [2] citation markers and inject clickable verification pills
  const renderFormattedText = (content: string) => {
    const parts = content.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citationId = parseInt(match[1], 10);
        const source = sources.find(s => s.citation_id === citationId);

        if (!source) return <span key={idx}>{part}</span>;

        const isVisual = source.source_type === 'visual';

        return (
          <button
            key={idx}
            onClick={() => onCitationClick(source)}
            className={`citation-token ${isVisual ? 'token-visual' : 'token-tabular'}`}
            title={
              isVisual
                ? `Verify Screenshot: ${source.file_name} (Page ${source.page_number})`
                : `Inspect Table: ${source.file_name} (${source.sheet_name})`
            }
          >
            {isVisual ? <Eye size={11} /> : <TableIcon size={11} />}
            <span>{citationId}</span>
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
        <div className="markdown-body">
          {isUser ? text : renderFormattedText(text)}
        </div>

        {/* Sources tray at the bottom of the assistant message */}
        {!isUser && sources.length > 0 && (
          <div className="sources-tray">
            <div className="sources-label">Verified Sources ({sources.length})</div>
            <div className="sources-cards">
              {sources.map(src => {
                const isVisual = src.source_type === 'visual';
                return (
                  <button
                    key={src.citation_id}
                    onClick={() => onCitationClick(src)}
                    className={`source-card ${isVisual ? 'visual' : 'tabular'}`}
                  >
                    <span className="card-index">[{src.citation_id}]</span>
                    <span>{src.file_name}</span>
                    <span className="card-meta">
                      {isVisual ? `Page ${src.page_number}` : `Sheet: ${src.sheet_name}`}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
