import React, { useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, X, Image as ImageIcon } from 'lucide-react';
import type { SourceNode } from '../types';

interface VisualModalProps {
  isOpen: boolean;
  onClose: () => void;
  source: SourceNode | null;
}

export const VisualModal: React.FC<VisualModalProps> = ({ isOpen, onClose, source }) => {
  const [scale, setScale] = useState<number>(1);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Reset zoom & pan when opening a new source
  useEffect(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, [source]);

  // Esc key listener to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !source) return null;

  const handleZoomIn = () => setScale(s => Math.min(s + 0.25, 4));
  const handleZoomOut = () => setScale(s => Math.max(s - 0.25, 0.5));
  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-sheet" onClick={e => e.stopPropagation()}>
        {/* Top bar */}
        <div className="modal-topbar">
          <div className="modal-doc-info">
            <ImageIcon size={16} color="#818cf8" />
            <strong>{source.file_name}</strong>
            <span className="page-chip">Page {source.page_number}</span>
          </div>

          <div className="modal-toolbar">
            <button onClick={handleZoomOut} className="tool-btn" title="Zoom Out"><ZoomOut size={14} /></button>
            <span className="tool-label">{Math.round(scale * 100)}%</span>
            <button onClick={handleZoomIn} className="tool-btn" title="Zoom In"><ZoomIn size={14} /></button>
            <button onClick={handleReset} className="tool-btn" title="Reset View"><RotateCcw size={14} /></button>
            <button onClick={onClose} className="tool-btn" title="Close (Esc)"><X size={15} /></button>
          </div>
        </div>

        {/* Pan and Zoom Canvas */}
        <div
          className="viewport-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
        >
          {source.screenshot_url ? (
            <img
              src={source.screenshot_url.startsWith('http') ? source.screenshot_url : `http://localhost:8000${source.screenshot_url}`}
              alt={`Page ${source.page_number} screenshot`}
              draggable={false}
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                transition: isDragging ? 'none' : 'transform 0.1s ease-out',
              }}
              className="viewport-img"
            />
          ) : (
            <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>
              No screenshot available for this document node.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
