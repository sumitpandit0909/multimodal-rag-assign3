export interface SourceNode {
  citation_id: number;
  file_name: string;
  source_type: 'visual' | 'tabular';
  screenshot_url?: string | null;
  page_number?: number | null;
  sheet_name?: string | null;
  raw_data?: Record<string, any>[] | null;
}

export interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  sources?: SourceNode[];
  timestamp: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources: SourceNode[];
}

export interface DocumentItem {
  file_name: string;
  source_type: 'visual' | 'tabular';
  chunks: number;
  screenshot_url?: string | null;
}

