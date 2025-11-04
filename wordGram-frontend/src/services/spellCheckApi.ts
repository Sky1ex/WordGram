import type { SpellCheckRequest, SpellCheckResponse } from '../types/spellCheck';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class SpellCheckApi {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async checkSpelling(text: string, language: string = 'ru'): Promise<SpellCheckResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/spell-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          language,
        } as SpellCheckRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SpellCheckResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Error checking spelling:', error);
      throw error;
    }
  }
}

export const spellCheckApi = new SpellCheckApi();
