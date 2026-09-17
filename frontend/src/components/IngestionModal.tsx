import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, FileText, FileSpreadsheet, Loader2, X, RefreshCw, Layers } from 'lucide-react';
import type { DocumentItem } from '../types';

interface IngestionModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiBaseUrl: string;
  onIngestionSuccess?: () => void;
}

export const IngestionModal: React.FC<IngestionModalProps> = ({
  isOpen,
  onClose,
  apiBaseUrl,
  onIngestionSuccess,
}) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'idle' | 'success' | 'error'; message: string }>({
    type: 'idle',
    message: '',
  });
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Esc key listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Fetch ingested documents on open
  const fetchDocuments = async () => {
    setIsLoadingDocs(true);
    try {
      const res = await fetch(`${apiBaseUrl}/documents`);
      const data = await res.json();
      if (data.documents) {
        setDocuments(data.documents);
      }
    } catch {
      // Ignored if offline
    } finally {
      setIsLoadingDocs(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchDocuments();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus({ type: 'idle', message: '' });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed with HTTP ${response.status}`);
      }

      const result = await response.json();
      setUploadStatus({
        type: 'success',
        message: result.message || `File ${selectedFile.name} successfully submitted to the ingestion pipeline!`,
      });
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';

      // Trigger callback and re-fetch documents after a short delay
      if (onIngestionSuccess) onIngestionSuccess();
      setTimeout(fetchDocuments, 3000);
    } catch (err: any) {
      setUploadStatus({
        type: 'error',
        message: `Ingestion error: ${err.message}`,
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-sheet ingestion-sheet" onClick={e => e.stopPropagation()}>
        {/* Topbar */}
        <div className="modal-topbar">
          <div className="modal-doc-info">
            <Layers size={18} color="#818cf8" />
            <strong>Ingestion Pipeline & Knowledge Base</strong>
            <span className="page-chip">Dual-Path: Visual & Excel</span>
          </div>
          <div className="modal-toolbar">
            <button onClick={fetchDocuments} className="tool-btn" title="Refresh Documents">
              <RefreshCw size={14} className={isLoadingDocs ? 'animate-spin' : ''} />
            </button>
            <button onClick={onClose} className="tool-btn" title="Close (Esc)">
              <X size={15} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="ingestion-body">
          {/* Upload Area */}
          <section className="upload-section">
            <h3 className="section-heading">Ingest New File</h3>
            <p className="section-subtext">
              Supports <strong>PDF, PPT, PPTX, DOCX</strong> (converted via headless LibreOffice & screenshot by PyMuPDF) and <strong>XLSX/XLS</strong> (parsed directly into tabular structured data).
            </p>

            <div
              className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${selectedFile ? 'dropzone-has-file' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.ppt,.pptx,.docx,.doc,.xlsx,.xls"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />

              <div className="dropzone-content">
                <UploadCloud size={32} className="dropzone-icon" />
                {selectedFile ? (
                  <div>
                    <p className="file-name-preview">{selectedFile.name}</p>
                    <span className="file-size-preview">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                ) : (
                  <div>
                    <p className="dropzone-prompt">Drag & drop your presentation, document, or spreadsheet here</p>
                    <span className="dropzone-sub">or click to browse from your computer</span>
                  </div>
                )}
              </div>
            </div>

            {/* Action Bar */}
            <div className="upload-actions">
              <button
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="ingest-btn"
              >
                {isUploading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    <span>Processing Ingestion Pipeline...</span>
                  </>
                ) : (
                  <>
                    <UploadCloud size={16} />
                    <span>Ingest File to MongoDB Atlas</span>
                  </>
                )}
              </button>
              {selectedFile && !isUploading && (
                <button onClick={() => setSelectedFile(null)} className="clear-btn">
                  Clear
                </button>
              )}
            </div>

            {/* Ingestion Status Alert */}
            {uploadStatus.message && (
              <div className={`status-banner banner-${uploadStatus.type}`}>
                {uploadStatus.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                <span>{uploadStatus.message}</span>
              </div>
            )}
          </section>

          {/* Documents Library */}
          <section className="library-section">
            <div className="library-header">
              <h3 className="section-heading">Indexed Documents in Atlas ({documents.length})</h3>
              <span className="live-indicator">● Synced with Vector Store</span>
            </div>

            {documents.length === 0 ? (
              <div className="empty-library">
                No documents found in the database. Drop files above to populate the knowledge base.
              </div>
            ) : (
              <div className="documents-grid">
                {documents.map((doc, idx) => {
                  const isVisual = doc.source_type === 'visual';
                  return (
                    <div key={idx} className="document-card">
                      <div className="doc-icon-wrapper">
                        {isVisual ? (
                          <FileText size={20} className="icon-visual" />
                        ) : (
                          <FileSpreadsheet size={20} className="icon-tabular" />
                        )}
                      </div>
                      <div className="doc-details">
                        <h4 className="doc-title" title={doc.file_name}>
                          {doc.file_name}
                        </h4>
                        <div className="doc-meta">
                          <span className={`type-badge ${isVisual ? 'badge-visual' : 'badge-tabular'}`}>
                            {isVisual ? 'Visual Pipeline' : 'Excel Tabular'}
                          </span>
                          <span className="chunks-badge">{doc.chunks} vectors</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
