import { useState, useEffect, useRef } from 'react';
import { spellCheckApi } from '../services/spellCheckApi';
import type { SpellError } from '../types/spellCheck';
import { useDebounce } from '../hooks/useDebounce';

interface SpellCheckerProps {
  language?: string;
}

export default function SpellChecker({ language = 'ru' }: SpellCheckerProps) {
  const [text, setText] = useState('');
  const [errors, setErrors] = useState<SpellError[]>([]);
  const [isChecking, setIsChecking] = useState(false);
  const [selectedError, setSelectedError] = useState<SpellError | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const debouncedText = useDebounce(text, 500);

  useEffect(() => {
    if (debouncedText.trim().length === 0) {
      setErrors([]);
      setIsChecking(false);
      return;
    }

    const checkSpelling = async () => {
      setIsChecking(true);
      try {
        const response = await spellCheckApi.checkSpelling(debouncedText, language);
        setErrors(response.errors || []);
      } catch (error) {
        console.error('Failed to check spelling:', error);
        // В случае ошибки API, просто очищаем ошибки
        setErrors([]);
      } finally {
        setIsChecking(false);
      }
    };

    checkSpelling();
  }, [debouncedText, language]);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    setSelectedError(null);
  };

  const handleSuggestionClick = (error: SpellError, suggestion: string) => {
    if (!textareaRef.current) return;

    const newText =
      text.slice(0, error.position.start) +
      suggestion +
      text.slice(error.position.end);

    setText(newText);
    setSelectedError(null);

    // Фокус на textarea после изменения
    setTimeout(() => {
      const newCursorPos = error.position.start + suggestion.length;
      textareaRef.current?.setSelectionRange(newCursorPos, newCursorPos);
      textareaRef.current?.focus();
    }, 0);
  };

  const highlightText = (text: string, errors: SpellError[]): React.ReactNode[] => {
    if (errors.length === 0) {
      return [text];
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    // Сортируем ошибки по позиции
    const sortedErrors = [...errors].sort((a, b) => a.position.start - b.position.start);

    sortedErrors.forEach((error, index) => {
      // Добавляем текст до ошибки
      if (error.position.start > lastIndex) {
        parts.push(text.slice(lastIndex, error.position.start));
      }

      // Добавляем слово с ошибкой
      const errorWord = text.slice(error.position.start, error.position.end);
      const isSelected = selectedError && 
        selectedError.position.start === error.position.start &&
        selectedError.position.end === error.position.end;
      parts.push(
        <span
          key={`error-${index}`}
          className={`bg-red-50 dark:bg-red-950/30 border-b-2 border-red-500 cursor-pointer transition-all duration-200 relative ${
            isSelected 
              ? 'bg-red-200 dark:bg-red-900/50 border-red-700 dark:border-red-500 shadow-md shadow-red-500/30' 
              : 'hover:bg-red-100 dark:hover:bg-red-900/40'
          }`}
          onClick={() => setSelectedError(error)}
          title={error.suggestions.join(', ')}
        >
          {errorWord}
        </span>
      );

      lastIndex = error.position.end;
    });

    // Добавляем оставшийся текст
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts;
  };

  const getErrorSuggestions = (error: SpellError | null) => {
    if (!error || error.suggestions.length === 0) return null;

    return (
      <div 
        className="bg-white dark:bg-gray-800 border-2 border-indigo-500 dark:border-indigo-400 rounded-lg p-6 mt-4 shadow-lg"
        style={{ animation: 'slideDown 0.3s ease-out' }}
      >
        <div className="flex items-center gap-4 mb-4 pb-3 border-b border-gray-200 dark:border-gray-700">
          <span className="font-semibold text-red-500 dark:text-red-400 text-lg">{error.word}</span>
          <span className="text-gray-600 dark:text-gray-400 text-sm">Предложения:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {error.suggestions.map((suggestion, index) => (
            <button
              key={index}
              className="bg-indigo-500 hover:bg-indigo-600 dark:bg-indigo-600 dark:hover:bg-indigo-700 text-white border-none px-5 py-2.5 rounded-md text-[0.95rem] font-medium cursor-pointer transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-indigo-500/30 active:translate-y-0 font-inherit"
              onClick={() => handleSuggestionClick(error, suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-8 md:p-4 font-sans">
      <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-6 pb-4 border-b-2 border-gray-200 dark:border-gray-700 gap-2">
        <h1 className="m-0 text-3xl md:text-2xl font-semibold text-gray-900 dark:text-white">Проверка орфографии</h1>
        {isChecking && (
          <span className="text-indigo-500 dark:text-indigo-400 text-sm font-medium animate-pulse">
            Проверка...
          </span>
        )}
        {!isChecking && errors.length > 0 && (
          <span className="bg-red-500 text-white px-3 py-1.5 rounded-full text-sm font-medium">
            Найдено ошибок: {errors.length}
          </span>
        )}
      </div>

      <div className="relative mb-6">
        <div className="relative bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden shadow-md transition-colors duration-300 focus-within:border-indigo-500 dark:focus-within:border-indigo-400">
          <textarea
            ref={textareaRef}
            className="w-full min-h-[400px] md:min-h-[300px] p-6 text-base leading-relaxed font-inherit border-none outline-none resize-y bg-transparent text-transparent caret-gray-900 dark:caret-white z-[2] relative"
            value={text}
            onChange={handleTextChange}
            placeholder="Введите текст для проверки орфографии..."
            spellCheck={false}
          />
          <div 
            className="absolute inset-0 p-6 text-base leading-relaxed font-inherit whitespace-pre-wrap break-words pointer-events-none z-[1] text-gray-900 dark:text-white"
            aria-hidden="true"
          >
            {highlightText(text, errors)}
          </div>
        </div>
      </div>

      {getErrorSuggestions(selectedError)}

      {errors.length > 0 && !selectedError && (
        <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-md text-center text-gray-600 dark:text-gray-400 text-sm">
          <p>Нажмите на выделенные слова, чтобы увидеть предложения по исправлению</p>
        </div>
      )}
    </div>
  );
}
