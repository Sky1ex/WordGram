import type { SpellCheckRequest, SpellCheckResponse } from '../types/spellCheck';

// Поддерживаем разные порты для бэкенда
// Если порт 8000 занят, сервер автоматически найдет свободный (8001, 8002, и т.д.)
// Укажите конкретный порт в .env файле, если нужно: VITE_API_URL=http://localhost:8001
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class SpellCheckApi {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	async checkSpelling(text: string, language: string = 'ru'): Promise<SpellCheckResponse> {
		const requestStartTime = Date.now();
		console.log('[SpellCheckApi] Sending spell check request:', {
			url: `${this.baseUrl}/api/spell-check`,
			textLength: text.length,
			language,
			timestamp: new Date().toISOString(),
		});

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

			const requestDuration = Date.now() - requestStartTime;

			if (!response.ok) {
				console.error('[SpellCheckApi] HTTP error:', {
					status: response.status,
					statusText: response.statusText,
					duration: `${requestDuration}ms`,
					timestamp: new Date().toISOString(),
				});
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const data: SpellCheckResponse = await response.json();
			console.log('[SpellCheckApi] Spell check response received:', {
				errorCount: data.errors?.length || 0,
				duration: `${requestDuration}ms`,
				timestamp: new Date().toISOString(),
			});
			return data;
		} catch (error) {
			const requestDuration = Date.now() - requestStartTime;
			console.error('[SpellCheckApi] Error checking spelling:', {
				error,
				duration: `${requestDuration}ms`,
				timestamp: new Date().toISOString(),
			});
			throw error;
		}
	}
}

export const spellCheckApi = new SpellCheckApi();
