export interface SpellError {
  word: string;
  position: {
    start: number;
    end: number;
  };
  suggestions: string[];
  severity?: 'error' | 'warning';
}

export interface SpellCheckResponse {
  errors: SpellError[];
  correctedText?: string;
}

export interface SpellCheckRequest {
  text: string;
  language?: string;
}
