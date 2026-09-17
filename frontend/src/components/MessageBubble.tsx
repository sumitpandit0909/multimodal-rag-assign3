import React from 'react';
import ReactMarkdown from 'react-markdown';
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

  // Parse [1], [2] citation markers and inject clickable verification buttons
  const renderCitationTokens = (content: string) => {
    const parts = content.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citationId = parseInt(match[1], 10);
        const source = sources.find(s => s.citation_id === citationId);

        if (!source) return part;

        const isVisual = source.source_type === 'visual';

        return (
          <button
            key={idx}
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onCitationClick(source);
            }}
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
      return part;
    });
  };

  const processChildren = (children: React.ReactNode): React.ReactNode => {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        return renderCitationTokens(child);
      }
      return child;
    });
  };

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
        <div className="markdown-body">
          {isUser ? (
            <div className="user-text-content">{text}</div>
          ) : (
            <ReactMarkdown
              components={{
                p: ({ children }) => <p>{processChildren(children)}</p>,
                li: ({ children }) => <li>{processChildren(children)}</li>,
                h1: ({ children }) => <h1 className="md-h1">{processChildren(children)}</h1>,
                h2: ({ children }) => <h2 className="md-h2">{processChildren(children)}</h2>,
                h3: ({ children }) => <h3 className="md-h3">{processChildren(children)}</h3>,
                h4: ({ children }) => <h4 className="md-h4">{processChildren(children)}</h4>,
                strong: ({ children }) => <strong>{processChildren(children)}</strong>,
                em: ({ children }) => <em>{processChildren(children)}</em>,
                blockquote: ({ children }) => <blockquote className="md-blockquote">{processChildren(children)}</blockquote>,
                table: ({ children }) => (
                  <div className="md-table-container">
                    <table className="md-table">{children}</table>
                  </div>
                ),
                code: ({ className, children, ...props }: any) => {
                  const isBlock = String(children).includes('\n');
                  if (!isBlock) {
                    return <code className="md-inline-code" {...props}>{children}</code>;
                  }
                  return (
                    <pre className="md-pre">
                      <code className={className} {...props}>{children}</code>
                    </pre>
                  );
                }
              }}
            >
              {text}
            </ReactMarkdown>
          )}
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
