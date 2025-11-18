// Content script для отслеживания инпутов и показа исправлений

(function () {
  'use strict';

  // Настройки
  const DEBOUNCE_DELAY = 1000; // Задержка перед проверкой (мс)
  const MIN_TEXT_LENGTH = 3; // Минимальная длина текста для проверки

  // Глобальное состояние
  let isEnabled = true;

  // Состояние для каждого инпута
  const inputStates = new Map();

  // Загрузка состояния включения/выключения
  chrome.storage.sync.get(['enabled'], (result) => {
    if (result.enabled !== undefined) {
      isEnabled = result.enabled;
    }
  });

  // Debounce функция
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Получение состояния для инпута
  function getInputState(element) {
    if (!inputStates.has(element)) {
      inputStates.set(element, {
        errors: [],
        isChecking: false,
        selectedError: null,
        debouncedCheck: null,
        suggestionsPanel: null,
        errorIndicator: null,
      });
    }
    return inputStates.get(element);
  }

  // Создание индикатора ошибок
  function createErrorIndicator(input) {
    const state = getInputState(input);
    if (state.errorIndicator) return state.errorIndicator;

    const indicator = document.createElement('div');
    indicator.className = 'wordgram-error-indicator';
    indicator.style.cssText = `
      position: absolute;
      right: 5px;
      top: 5px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgb(239, 68, 68);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      color: rgb(239, 68, 68);
      pointer-events: none;
      z-index: 1000;
      display: none;
    `;

    // Позиционируем относительно инпута
    const container = input.parentElement;
    if (container.style.position === '' || container.style.position === 'static') {
      container.style.position = 'relative';
    }
    container.appendChild(indicator);

    state.errorIndicator = indicator;
    return indicator;
  }

  // Экранирование HTML
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Проверка доступности расширения
  function isExtensionAvailable() {
    try {
      return chrome.runtime && chrome.runtime.id;
    } catch (e) {
      return false;
    }
  }

  // Проверка орфографии
  async function checkSpelling(text, language = 'ru') {
    return new Promise((resolve, reject) => {
      // Проверяем доступность расширения
      if (!isExtensionAvailable()) {
        reject(new Error('Extension context invalidated. Please reload the page.'));
        return;
      }

      chrome.runtime.sendMessage(
        {
          action: 'checkSpelling',
          text,
          language,
        },
        (response) => {
          // Проверяем ошибку runtime перед проверкой response
          if (chrome.runtime.lastError) {
            const errorMsg = chrome.runtime.lastError.message;
            // Если контекст расширения недействителен, предлагаем перезагрузить страницу
            if (errorMsg.includes('Extension context invalidated') || 
                errorMsg.includes('message port closed')) {
              console.warn('[WordGram] Extension context invalidated. Please reload the page to continue using WordGram.');
              reject(new Error('Extension context invalidated. Please reload the page.'));
            } else {
              reject(new Error(errorMsg));
            }
            return;
          }

          // Проверяем, что response существует
          if (!response) {
            reject(new Error('No response from extension'));
            return;
          }

          if (response.success) {
            resolve(response.data);
          } else {
            reject(new Error(response.error || 'Unknown error'));
          }
        }
      );
    });
  }

  // Обработка проверки текста
  async function handleTextCheck(input, text) {
    if (!isEnabled) {
      return;
    }

    const state = getInputState(input);

    if (text.trim().length < MIN_TEXT_LENGTH) {
      state.errors = [];
      updateErrorDisplay(input);
      return;
    }

    state.isChecking = true;
    updateErrorDisplay(input);

    try {
      const result = await checkSpelling(text, 'ru');
      state.errors = result.errors || [];

      // Очищаем выбранную ошибку, если она больше не существует
      if (state.selectedError) {
        const stillExists = state.errors.find(
          (err) =>
            err.position.start === state.selectedError.position.start &&
            err.position.end === state.selectedError.position.end
        );
        if (!stillExists) {
          state.selectedError = null;
          hideSuggestionsPanel(input);
        }
      }

      updateErrorDisplay(input);
    } catch (error) {
      // Не логируем ошибку "Extension context invalidated" как критическую
      if (error.message && error.message.includes('Extension context invalidated')) {
        console.warn('[WordGram] Extension was reloaded. Please reload the page to continue.');
      } else {
        console.error('[WordGram] Error checking spelling:', error);
      }
      state.errors = [];
      state.selectedError = null;
      updateErrorDisplay(input);
    } finally {
      state.isChecking = false;
    }
  }

  // Обновление отображения ошибок
  function updateErrorDisplay(input) {
    const state = getInputState(input);
    
    // Получаем текст в зависимости от типа элемента
    let text = '';
    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      text = input.value || '';
    } else if (input.isContentEditable) {
      text = input.innerText || input.textContent || '';
    } else {
      text = input.textContent || '';
    }

    // Скрываем индикатор ошибок (количество будет в панели)
    const indicator = createErrorIndicator(input);
    indicator.style.display = 'none';
    
    if (state.errors.length > 0) {
      // Добавляем класс для подчеркивания
      input.classList.add('wordgram-has-errors');
    } else {
      input.classList.remove('wordgram-has-errors');
    }

    // Если есть выбранная ошибка, показываем панель
    if (state.selectedError && state.errors.length > 0) {
      // Проверяем, что выбранная ошибка все еще существует
      const errorStillExists = state.errors.find(
        err => err.position.start === state.selectedError.position.start &&
                err.position.end === state.selectedError.position.end
      );
      
      if (errorStillExists) {
        showSuggestionsPanel(input, state.selectedError);
      } else {
        // Если ошибка больше не существует, выбираем первую
        if (state.errors.length > 0) {
          state.selectedError = state.errors[0];
          showSuggestionsPanel(input, state.selectedError);
        } else {
          state.selectedError = null;
          hideSuggestionsPanel(input);
        }
      }
    } else if (state.errors.length > 0 && !state.selectedError) {
      // Если есть ошибки, но нет выбранной - выбираем первую
      state.selectedError = state.errors[0];
      showSuggestionsPanel(input, state.selectedError);
    } else {
      hideSuggestionsPanel(input);
    }
  }

  // Нахождение ошибки по позиции курсора
  function findErrorAtPosition(input, cursorPos) {
    const state = getInputState(input);
    
    // Находим ошибку, которая содержит позицию курсора
    const error = state.errors.find(err => {
      return cursorPos >= err.position.start && cursorPos <= err.position.end;
    });

    return error || null;
  }

  // Показ панели с предложениями справа от инпута
  function showSuggestionsPanel(input, error) {
    const state = getInputState(input);

    // Удаляем существующую панель
    if (state.suggestionsPanel) {
      state.suggestionsPanel.remove();
    }

    if (!error || !error.suggestions || error.suggestions.length === 0) {
      return;
    }

    // Получаем общее количество ошибок
    const totalErrors = state.errors.length;
    const currentErrorIndex = state.errors.findIndex(
      err => err.position.start === error.position.start && err.position.end === error.position.end
    ) + 1;

    // Создаем панель
    const panel = document.createElement('div');
    panel.className = 'wordgram-suggestions-panel';
    panel.innerHTML = `
      <div class="wordgram-suggestions-header">
        <div class="wordgram-suggestions-title">
          <span class="wordgram-suggestions-word">${escapeHtml(error.word)}</span>
          <span class="wordgram-errors-count">Ошибка ${currentErrorIndex} из ${totalErrors}</span>
        </div>
        <button class="wordgram-close-btn" title="Закрыть">×</button>
      </div>
      <div class="wordgram-suggestions-label">Предложения:</div>
      <div class="wordgram-suggestions-list">
        ${error.suggestions
          .map(
            (suggestion, index) =>
              `<button class="wordgram-suggestion-btn" data-suggestion="${escapeHtml(suggestion)}">${escapeHtml(suggestion)}</button>`
          )
          .join('')}
      </div>
    `;

    // Обработка закрытия
    const closeBtn = panel.querySelector('.wordgram-close-btn');
    closeBtn.addEventListener('click', () => {
      state.selectedError = null;
      hideSuggestionsPanel(input);
      updateErrorDisplay(input);
    });

    // Обработка клика по предложению
    panel.querySelectorAll('.wordgram-suggestion-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const suggestion = btn.dataset.suggestion;
        applySuggestion(input, error, suggestion);
      });
    });

    document.body.appendChild(panel);
    state.suggestionsPanel = panel;

    // Позиционируем панель с учетом границ viewport
    positionPanel(panel, input);
  }

  // Позиционирование панели с учетом границ viewport
  function positionPanel(panel, input) {
    const rect = input.getBoundingClientRect();
    const padding = 10; // Отступ от инпута
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Сначала устанавливаем базовую позицию справа (fixed позиционирование использует координаты viewport)
    panel.style.position = 'fixed';
    panel.style.left = `${rect.right + padding}px`;
    panel.style.top = `${rect.top}px`;
    panel.style.zIndex = '10000';
    
    // Ждем, пока панель отрисуется, чтобы получить её размеры
    setTimeout(() => {
      const panelRect = panel.getBoundingClientRect();
      const panelWidth = panelRect.width;
      const panelHeight = panelRect.height;
      
      let left = rect.right + padding;
      let top = rect.top;
      
      // Проверяем правую границу viewport
      if (left + panelWidth > viewportWidth) {
        // Пробуем слева от инпута
        const leftPosition = rect.left - panelWidth - padding;
        if (leftPosition >= 0) {
          left = leftPosition;
        } else {
          // Если слева не помещается, прижимаем к правому краю viewport
          left = Math.max(padding, viewportWidth - panelWidth - padding);
        }
      }
      
      // Проверяем нижнюю границу viewport
      if (top + panelHeight > viewportHeight) {
        // Поднимаем панель вверх, чтобы она поместилась
        top = Math.max(padding, viewportHeight - panelHeight - padding);
      }
      
      // Проверяем верхнюю границу viewport
      if (top < 0) {
        top = padding;
      }
      
      // Проверяем левую границу viewport
      if (left < 0) {
        left = padding;
      }
      
      // Убеждаемся, что панель не выходит за правую границу
      if (left + panelWidth > viewportWidth) {
        left = viewportWidth - panelWidth - padding;
      }
      
      // Убеждаемся, что панель не выходит за нижнюю границу
      if (top + panelHeight > viewportHeight) {
        top = viewportHeight - panelHeight - padding;
      }
      
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
    }, 0);
  }

  // Скрытие панели с предложениями
  function hideSuggestionsPanel(input) {
    const state = getInputState(input);
    if (state.suggestionsPanel) {
      state.suggestionsPanel.remove();
      state.suggestionsPanel = null;
    }
  }

  // Применение предложения
  function applySuggestion(input, error, suggestion) {
    const state = getInputState(input);
    
    // Получаем текст в зависимости от типа элемента
    let text = '';
    let isContentEditable = false;
    
    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      text = input.value || '';
    } else if (input.isContentEditable) {
      text = input.innerText || input.textContent || '';
      isContentEditable = true;
    } else {
      text = input.textContent || '';
    }

    // Заменяем слово
    const newText =
      text.slice(0, error.position.start) +
      suggestion +
      text.slice(error.position.end);

    // Обновляем значение инпута
    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      input.value = newText;
    } else if (isContentEditable) {
      // Для contenteditable сохраняем структуру, заменяя только текст
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(input);
      range.setStart(input, 0);
      range.setEnd(input, input.childNodes.length || 0);
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand('insertText', false, newText);
    } else {
      input.textContent = newText;
    }

    // Триггерим событие изменения
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));

    // Очищаем выбранную ошибку
    state.selectedError = null;
    hideSuggestionsPanel(input);

    // Пересчитываем позиции остальных ошибок
    const delta = suggestion.length - (error.position.end - error.position.start);
    state.errors = state.errors
      .filter(
        (e) =>
          !(
            e.position.start === error.position.start &&
            e.position.end === error.position.end
          )
      )
      .map((e) => {
        if (e.position.start >= error.position.end) {
          return {
            ...e,
            position: {
              start: e.position.start + delta,
              end: e.position.end + delta,
            },
          };
        }
        return e;
      });

    // Запускаем проверку заново
    const debouncedCheck = debounce(() => {
      handleTextCheck(input, newText);
    }, DEBOUNCE_DELAY);
    debouncedCheck();
  }

  // Обработка клика по инпуту для выбора ошибки
  function handleInputClick(input, event) {
    if (!isEnabled) return;

    const state = getInputState(input);
    if (state.errors.length === 0) {
      state.selectedError = null;
      hideSuggestionsPanel(input);
      return;
    }

    // Получаем позицию курсора
    let cursorPos = 0;
    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      cursorPos = input.selectionStart || 0;
    } else if (input.isContentEditable) {
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const preCaretRange = range.cloneRange();
        preCaretRange.selectNodeContents(input);
        preCaretRange.setEnd(range.endContainer, range.endOffset);
        cursorPos = preCaretRange.toString().length;
      }
    }

    // Находим ошибку в позиции курсора
    const error = findErrorAtPosition(input, cursorPos);
    
    if (error) {
      state.selectedError = error;
      updateErrorDisplay(input);
    } else {
      // Если клик не на ошибке, но рядом - находим ближайшую
      const closestError = state.errors.reduce((closest, err) => {
        const distance = Math.min(
          Math.abs(cursorPos - err.position.start),
          Math.abs(cursorPos - err.position.end)
        );
        const closestDistance = closest ? Math.min(
          Math.abs(cursorPos - closest.position.start),
          Math.abs(cursorPos - closest.position.end)
        ) : Infinity;
        return distance < closestDistance ? err : closest;
      }, null);

      if (closestError) {
        const distance = Math.min(
          Math.abs(cursorPos - closestError.position.start),
          Math.abs(cursorPos - closestError.position.end)
        );
        // Если ошибка в пределах 5 символов, показываем её
        if (distance <= 5) {
          state.selectedError = closestError;
          updateErrorDisplay(input);
        } else {
          state.selectedError = null;
          hideSuggestionsPanel(input);
        }
      } else {
        state.selectedError = null;
        hideSuggestionsPanel(input);
      }
    }
  }

  // Обработка выделения текста
  function handleInputSelect(input) {
    if (!isEnabled) return;

    const state = getInputState(input);
    if (state.errors.length === 0) return;

    let startPos = 0;
    let endPos = 0;

    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      startPos = input.selectionStart || 0;
      endPos = input.selectionEnd || 0;
    } else if (input.isContentEditable) {
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const preStartRange = range.cloneRange();
        preStartRange.selectNodeContents(input);
        preStartRange.setEnd(range.startContainer, range.startOffset);
        startPos = preStartRange.toString().length;
        
        const preEndRange = range.cloneRange();
        preEndRange.selectNodeContents(input);
        preEndRange.setEnd(range.endContainer, range.endOffset);
        endPos = preEndRange.toString().length;
      }
    }

    // Находим ошибку в выделенном диапазоне
    const error = state.errors.find(err => {
      return (
        (startPos >= err.position.start && startPos <= err.position.end) ||
        (endPos >= err.position.start && endPos <= err.position.end) ||
        (err.position.start >= startPos && err.position.end <= endPos)
      );
    });

    if (error) {
      state.selectedError = error;
      updateErrorDisplay(input);
    }
  }

  // Инициализация инпута
  function initializeInput(input) {
    const state = getInputState(input);

    // Создаем debounced функцию проверки
    state.debouncedCheck = debounce((text) => {
      handleTextCheck(input, text);
    }, DEBOUNCE_DELAY);

    // Обработчик изменения текста
    const handleInput = (e) => {
      let text = '';
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        text = e.target.value || '';
      } else if (e.target.isContentEditable) {
        text = e.target.innerText || e.target.textContent || '';
      } else {
        text = e.target.textContent || '';
      }
      state.selectedError = null;
      hideSuggestionsPanel(input);
      state.debouncedCheck(text);
    };

    input.addEventListener('input', handleInput);
    input.addEventListener('paste', handleInput);
    
    // Обработчик клика для выбора ошибки
    input.addEventListener('click', (e) => {
      setTimeout(() => handleInputClick(input, e), 10);
    });

    // Обработчик выделения
    input.addEventListener('select', () => {
      setTimeout(() => handleInputSelect(input), 10);
    });

    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {
      input.addEventListener('selectionchange', () => {
        setTimeout(() => handleInputSelect(input), 10);
      });
    }

    // Обработчик изменения размера
    const resizeObserver = new ResizeObserver(() => {
      updateErrorDisplay(input);
    });
    resizeObserver.observe(input);
  }

  // Поиск всех инпутов на странице
  function findInputs() {
    const inputs = document.querySelectorAll(
      'input[type="text"], input[type="email"], input[type="search"], input[type="url"], textarea, [contenteditable="true"]'
    );
    return Array.from(inputs);
  }

  // Инициализация всех инпутов
  function initializeAllInputs() {
    const inputs = findInputs();
    inputs.forEach((input) => {
      if (!inputStates.has(input)) {
        initializeInput(input);
      }
    });
  }

  // Обработка динамически добавленных инпутов
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          // Проверяем сам узел
          if (
            node.matches &&
            node.matches(
              'input[type="text"], input[type="email"], input[type="search"], input[type="url"], textarea, [contenteditable="true"]'
            )
          ) {
            initializeInput(node);
          }

          // Проверяем дочерние элементы
          const inputs = node.querySelectorAll
            ? node.querySelectorAll(
                'input[type="text"], input[type="email"], input[type="search"], input[type="url"], textarea, [contenteditable="true"]'
              )
            : [];
          inputs.forEach((input) => {
            initializeInput(input);
          });
        }
      });
    });
  });

  // Запуск при загрузке страницы
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initializeAllInputs();
      observer.observe(document.body, {
        childList: true,
        subtree: true,
      });
    });
  } else {
    initializeAllInputs();
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  // Обработка клика вне панели предложений
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.wordgram-suggestions-panel')) {
      inputStates.forEach((state, input) => {
        if (state.selectedError && !input.contains(e.target)) {
          state.selectedError = null;
          updateErrorDisplay(input);
          hideSuggestionsPanel(input);
        }
      });
    }
  });

  // Обработка сообщений от popup (после определения всех функций)
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'toggle') {
      isEnabled = request.enabled;
      if (!isEnabled) {
        // Очищаем все ошибки при отключении
        inputStates.forEach((state, input) => {
          state.errors = [];
          state.selectedError = null;
          updateErrorDisplay(input);
          hideSuggestionsPanel(input);
        });
      }
      sendResponse({ success: true });
    }
  });
})();
