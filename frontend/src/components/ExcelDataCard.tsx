import React, { useEffect } from 'react';
import { Table as TableIcon, X } from 'lucide-react';
import type { SourceNode } from '../types';

interface ExcelDataCardProps {
  isOpen: boolean;
  onClose: () => void;
  source: SourceNode | null;
}

export const ExcelDataCard: React.FC<ExcelDataCardProps> = ({ isOpen, onClose, source }) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !source) return null;

  const rows = source.raw_data || [];
  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-sheet" onClick={e => e.stopPropagation()}>
        {/* Topbar */}
        <div className="modal-topbar">
          <div className="modal-doc-info">
            <TableIcon size={16} color="#34d399" />
            <strong>{source.file_name}</strong>
            <span className="page-chip">Sheet: {source.sheet_name}</span>
          </div>
          <div className="modal-toolbar">
            <button onClick={onClose} className="tool-btn" title="Close (Esc)"><X size={15} /></button>
          </div>
        </div>

        {/* Tabular data table */}
        <div className="excel-table-container">
          {rows.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '3rem' }}>
              No structured row preview available for this table.
            </div>
          ) : (
            <table className="minimal-table">
              <thead>
                <tr>
                  {columns.map(col => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {columns.map(col => (
                      <td key={col}>{String(row[col] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};
