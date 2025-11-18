// Background service worker для обработки API запросов

const DEFAULT_API_URL = 'http://localhost:8000';

// Получаем URL API из настроек
async function getApiUrl() {
  const result = await chrome.storage.sync.get(['apiUrl']);
  return result.apiUrl || DEFAULT_API_URL;
}

// Сохраняем URL API в настройках
async function setApiUrl(url) {
  await chrome.storage.sync.set({ apiUrl: url });
}

// Проверка орфографии через API
async function checkSpelling(text, language = 'ru') {
  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/api/spell-check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        language,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('[WordGram] Error checking spelling:', error);
    throw error;
  }
}

// Обработка сообщений от content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkSpelling') {
    checkSpelling(request.text, request.language)
      .then((result) => {
        sendResponse({ success: true, data: result });
      })
      .catch((error) => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Асинхронный ответ
  }

  if (request.action === 'getApiUrl') {
    getApiUrl().then((url) => {
      sendResponse({ apiUrl: url });
    });
    return true;
  }

  if (request.action === 'setApiUrl') {
    setApiUrl(request.apiUrl).then(() => {
      sendResponse({ success: true });
    });
    return true;
  }
});

// Проверка доступности API при установке
chrome.runtime.onInstalled.addListener(async () => {
  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/`);
    if (response.ok) {
      console.log('[WordGram] API доступен:', apiUrl);
    }
  } catch (error) {
    console.warn('[WordGram] API недоступен:', error.message);
  }
});

