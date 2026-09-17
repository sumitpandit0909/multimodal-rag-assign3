import React, { useState, useEffect, useRef } from 'react';
import { 
  UploadCloud, CheckCircle2, AlertCircle, FileText, FileSpreadsheet, 
  Loader2, X, RefreshCw, Layers, ArrowRight, AlertTriangle, Sparkles 
} from 'lucide-react';
import type { DocumentItem } from '../types';

interface IngestionStage {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  detail?: string;
}

interface IngestionJob {
  job_id: string;
  filename: string;
  status: 'processing' | 'completed' | 'failed';
  current_stage: string;
  progress_percent: number;
  stages: IngestionStage[];
  chunks_indexed: number;
  error?: string | null;
  error_details?: string | null;
  started_at: number;
  completed_at?: number | null;
}

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
  const [activeJob, setActiveJob] = useState<IngestionJob | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<any>(null);

  // Esc key listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

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

  // Poll active job status
  const startPollingJob = (jobId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/jobs/${jobId}`);
        if (!res.ok) return;
        const jobData: IngestionJob = await res.json();
        setActiveJob(jobData);

        if (jobData.status === 'completed') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          fetchDocuments();
          if (onIngestionSuccess) onIngestionSuccess();
        } else if (jobData.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } catch (err) {
        console.error('Job polling error:', err);
      }
    }, 800);
  };

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
    setActiveJob(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText}`);
      }

      const result = await response.json();
      if (result.job_id) {
        startPollingJob(result.job_id);
      }
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setActiveJob({
        job_id: 'err-' + Date.now(),
        filename: selectedFile.name,
        status: 'failed',
        current_stage: 'Upload Request',
        progress_percent: 0,
        stages: [
          { id: 'upload', name: 'File Upload & Validation', status: 'failed', detail: err.message }
        ],
        chunks_indexed: 0,
        error: err.message,
        started_at: Date.now() / 1000
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleResetJob = () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    setActiveJob(null);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-sheet ingestion-sheet" onClick={e => e.stopPropagation()}>
        {/* Topbar */}
        <div className="modal-topbar">
          <div className="modal-doc-info">
            <Layers size={18} color="#818cf8" />
            <strong>Ingestion Pipeline & Knowledge Base</strong>
            <span className="page-chip">Gemma-3-27b-it Vision & LangSmith</span>
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
          {/* Upload & Pipeline Progress Area */}
          <section className="upload-section">
            <div className="section-title-row">
              <h3 className="section-heading">Ingest New Document</h3>
              <span className="pipeline-pill">
                <Sparkles size={12} /> Powered by google/gemma-3-27b-it
              </span>
            </div>
            <p className="section-subtext">
              <strong>Path A:</strong> PDF / PPT / DOCX &rarr; Headless LibreOffice &rarr; PyMuPDF 150 DPI &rarr; Gemma-3-27b-it Vision &rarr; LlamaParse &rarr; Atlas Vector Store<br />
              <strong>Path B:</strong> Excel (XLSX/XLS) &rarr; Structured Rows &amp; Sheets &rarr; Atlas (strictly zero screenshots)
            </p>

            {/* If there is an active job, show the live stage tracker */}
            {activeJob ? (
              <div className={`live-pipeline-card status-${activeJob.status}`}>
                <div className="pipeline-card-header">
                  <div className="file-info-header">
                    <span className="active-file-title">{activeJob.filename}</span>
                    <span className={`status-pill pill-${activeJob.status}`}>
                      {activeJob.status === 'processing' && <Loader2 size={12} className="animate-spin" />}
                      {activeJob.status === 'completed' && <CheckCircle2 size={12} />}
                      {activeJob.status === 'failed' && <AlertCircle size={12} />}
                      {activeJob.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="header-actions">
                    {(activeJob.status === 'completed' || activeJob.status === 'failed') && (
                      <button onClick={handleResetJob} className="action-pill-btn">
                        Upload Another Document
                      </button>
                    )}
                  </div>
                </div>

                {/* Progress bar */}
                <div className="pipeline-progress-track">
                  <div 
                    className={`pipeline-progress-fill ${activeJob.status === 'failed' ? 'progress-fill-failed' : ''}`}
                    style={{ width: `${activeJob.progress_percent}%` }}
                  />
                </div>
                <div className="progress-label-row">
                  <span className="current-stage-text">
                    <strong>Stage:</strong> {activeJob.current_stage}
                  </span>
                  <span className="percent-text">{activeJob.progress_percent}%</span>
                </div>

                {/* Stages List */}
                <div className="stages-stepper">
                  {activeJob.stages.map((stage, idx) => {
                    const isPending = stage.status === 'pending';
                    const isInProgress = stage.status === 'in_progress';
                    const isCompleted = stage.status === 'completed';
                    const isFailed = stage.status === 'failed';

                    return (
                      <div key={idx} className={`stage-step-item step-${stage.status}`}>
                        <div className="step-bullet">
                          {isCompleted && <CheckCircle2 size={15} className="step-icon-success" />}
                          {isInProgress && <Loader2 size={15} className="step-icon-active animate-spin" />}
                          {isPending && <div className="step-icon-pending" />}
                          {isFailed && <AlertCircle size={15} className="step-icon-failed" />}
                        </div>
                        <div className="step-content">
                          <div className="step-name-row">
                            <span className="step-name">{stage.name}</span>
                            <span className={`step-badge badge-${stage.status}`}>{stage.status}</span>
                          </div>
                          {stage.detail && <p className="step-detail">{stage.detail}</p>}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Completed Banner */}
                {activeJob.status === 'completed' && (
                  <div className="job-completion-alert">
                    <CheckCircle2 size={18} color="#34d399" />
                    <div>
                      <strong>Ingestion Succeeded!</strong>
                      <p>All {activeJob.chunks_indexed} document chunks were indexed into MongoDB Atlas and are immediately ready for queries.</p>
                    </div>
                  </div>
                )}

                {/* Failure Error Alert */}
                {activeJob.status === 'failed' && (
                  <div className="job-failure-alert">
                    <AlertTriangle size={20} color="#f87171" className="failure-icon" />
                    <div className="failure-details">
                      <strong>Ingestion Pipeline Failed</strong>
                      <p className="failure-message">{activeJob.error || 'An unexpected error occurred during ingestion.'}</p>
                      {activeJob.error_details && (
                        <details className="failure-accordion">
                          <summary>View Traceback Details</summary>
                          <pre>{activeJob.error_details}</pre>
                        </details>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Dropzone for new uploads */
              <>
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
                        <span>Submitting to Pipeline...</span>
                      </>
                    ) : (
                      <>
                        <UploadCloud size={16} />
                        <span>Start Ingestion Pipeline</span>
                        <ArrowRight size={14} />
                      </>
                    )}
                  </button>
                  {selectedFile && !isUploading && (
                    <button onClick={() => setSelectedFile(null)} className="clear-btn">
                      Clear
                    </button>
                  )}
                </div>
              </>
            )}
          </section>

          {/* Documents Library */}
          <section className="library-section">
            <div className="library-header">
              <h3 className="section-heading">Indexed Documents in Atlas ({documents.length})</h3>
              <span className="live-indicator">● Live Vector Store</span>
            </div>

            {documents.length === 0 ? (
              <div className="empty-library">
                No documents found in the database. Ingest documents above to populate the enterprise knowledge base.
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
