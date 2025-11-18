// Popup script для настроек расширения

document.addEventListener('DOMContentLoaded', async () => {
  const apiUrlInput = document.getElementById('apiUrl');
  const saveApiUrlBtn = document.getElementById('saveApiUrl');
  const apiStatus = document.getElementById('apiStatus');
  const enableToggle = document.getElementById('enableToggle');
  const statusMessage = document.getElementById('statusMessage');

  // Загрузка текущих настроек
  async function loadSettings() {
    // Загружаем URL API
    chrome.runtime.sendMessage({ action: 'getApiUrl' }, (response) => {
      if (response && response.apiUrl) {
        apiUrlInput.value = response.apiUrl;
      }
    });

    // Загружаем состояние включения/выключения
    const result = await chrome.storage.sync.get(['enabled']);
    if (result.enabled !== false) {
      enableToggle.classList.add('active');
    }
  }

  // Сохранение URL API
  saveApiUrlBtn.addEventListener('click', async () => {
    const apiUrl = apiUrlInput.value.trim();
    
    if (!apiUrl) {
      showStatus(apiStatus, 'Пожалуйста, укажите URL API', 'error');
      return;
    }

    // Проверяем формат URL
    try {
      new URL(apiUrl);
    } catch (e) {
      showStatus(apiStatus, 'Неверный формат URL', 'error');
      return;
    }

    // Сохраняем URL
    chrome.runtime.sendMessage(
      { action: 'setApiUrl', apiUrl },
      async (response) => {
        if (response && response.success) {
          showStatus(apiStatus, 'URL сохранен', 'success');
          
          // Проверяем доступность API
          try {
            const testResponse = await fetch(`${apiUrl}/`);
            if (testResponse.ok) {
              showStatus(apiStatus, 'API доступен и работает', 'success');
            } else {
              showStatus(apiStatus, 'API недоступен (проверьте сервер)', 'error');
            }
          } catch (error) {
            showStatus(apiStatus, 'Не удалось подключиться к API', 'error');
          }
        } else {
          showStatus(apiStatus, 'Ошибка при сохранении', 'error');
        }
      }
    );
  });

  // Переключение включения/выключения
  enableToggle.addEventListener('click', async () => {
    const isActive = enableToggle.classList.contains('active');
    
    if (isActive) {
      enableToggle.classList.remove('active');
      await chrome.storage.sync.set({ enabled: false });
      showStatus(statusMessage, 'Проверка орфографии отключена', 'info');
      
      // Отправляем сообщение content script для отключения
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'toggle', enabled: false });
      });
    } else {
      enableToggle.classList.add('active');
      await chrome.storage.sync.set({ enabled: true });
      showStatus(statusMessage, 'Проверка орфографии включена', 'success');
      
      // Отправляем сообщение content script для включения
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'toggle', enabled: true });
      });
    }
  });

  // Показ статуса
  function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status ${type}`;
    element.classList.remove('hidden');
    
    setTimeout(() => {
      element.classList.add('hidden');
    }, 3000);
  }

  // Загружаем настройки при открытии popup
  loadSettings();
});

