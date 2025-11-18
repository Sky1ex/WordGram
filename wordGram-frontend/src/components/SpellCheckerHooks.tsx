import React from 'react';
import { spellCheckApi } from "../services/spellCheckApi";
import type { SpellError } from '../types/spellCheck';

interface CheckSpellingProps {
    setIsChecking: React.Dispatch<React.SetStateAction<boolean>>;
    debouncedText: string;
    language?: string;
    setErrors: React.Dispatch<React.SetStateAction<SpellError[]>>;
    selectedError: SpellError | null;
    setSelectedError: React.Dispatch<React.SetStateAction<SpellError | null>>;
}

interface HandleTextChangeProps {
    e: React.ChangeEvent<HTMLTextAreaElement>;
    text: string;
    setText: React.Dispatch<React.SetStateAction<string>>;
    setSelectedError: React.Dispatch<React.SetStateAction<SpellError | null>>;
}

interface HandleSuggestionClickProps {
    error: SpellError;
    suggestion: string;
    text: string;
    setText: React.Dispatch<React.SetStateAction<string>>;
    setErrors: React.Dispatch<React.SetStateAction<SpellError[]>>;
    setSelectedError: React.Dispatch<React.SetStateAction<SpellError | null>>;
    textareaRef: React.RefObject<HTMLTextAreaElement | null>;
    markSkipNextCheck: () => void;
}

interface HighlightTextProps {
    text: string;
    errors: SpellError[];
    selectedError: SpellError | null;
}

interface HandleTextareaClickProps {
    errors: SpellError[];
    text: string;
    textareaRef: React.RefObject<HTMLTextAreaElement | null>;
    setSelectedError: React.Dispatch<React.SetStateAction<SpellError | null>>;
}

interface GetErrorSuggestionsProps {
    error: SpellError | null;
    text: string;
    debouncedText: string;
    handleSuggestionClick: (error: SpellError, suggestion: string) => void;
}

export const checkSpelling = async ({
    setIsChecking,
    debouncedText,
    language,
    setErrors,
    selectedError,
    setSelectedError
}: CheckSpellingProps) => {
    setIsChecking(true);
    console.log('[SpellChecker] Starting spell check:', {
        textLength: debouncedText.length,
        language,
        timestamp: new Date().toISOString(),
    });
    try {
        const response = await spellCheckApi.checkSpelling(debouncedText, language);
        const errorCount = response.errors?.length || 0;
        console.log('[SpellChecker] Spell check completed:', {
            errorCount,
            errors: response.errors?.map(e => ({
                word: e.word,
                position: e.position,
                suggestionsCount: e.suggestions.length,
            })),
            timestamp: new Date().toISOString(),
        });
        const newErrors = response.errors || [];
        setErrors(newErrors);

        // Очищаем выбранную ошибку, если она больше не существует или не соответствует тексту
        if (selectedError) {
            const stillExists = newErrors.find(err =>
                err.position.start === selectedError.position.start &&
                err.position.end === selectedError.position.end
            );

            if (!stillExists) {
                setSelectedError(null);
            } else {
                // Проверяем, что слово все еще соответствует
                const actualWord = debouncedText.slice(stillExists.position.start, stillExists.position.end);
                if (actualWord !== stillExists.word && actualWord.toLowerCase() !== stillExists.word.toLowerCase()) {
                    setSelectedError(null);
                }
            }
        }
    } catch (error) {
        console.error('[SpellChecker] Failed to check spelling:', {
            error,
            textLength: debouncedText.length,
            language,
            timestamp: new Date().toISOString(),
        });
        // В случае ошибки API, просто очищаем ошибки
        setErrors([]);
    } finally {
        setIsChecking(false);
    }
};

export const handleTextChange = ({
    e,
    text,
    setText,
    setSelectedError
}: HandleTextChangeProps) => {
    const newText = e.target.value;
    console.log('[SpellChecker] Text changed:', {
        length: newText.length,
        previousLength: text.length,
        timestamp: new Date().toISOString(),
    });
    setText(newText);
    setSelectedError(null);
};

export const handleSuggestionClick = ({
    error,
    suggestion,
    text,
    setText,
    setErrors,
    setSelectedError,
    textareaRef,
    markSkipNextCheck
}: HandleSuggestionClickProps) => {
    if (!textareaRef.current) return;

    console.log('[SpellChecker] Suggestion clicked:', {
        originalWord: error.word,
        suggestion,
        position: error.position,
        timestamp: new Date().toISOString(),
    });

    // Пропускаем следующую автоматическую проверку после применения предложения
    markSkipNextCheck();

    const newText =
        text.slice(0, error.position.start) +
        suggestion +
        text.slice(error.position.end);

    setText(newText);
    setSelectedError(null);

    // Удаляем исправленную ошибку и сдвигаем позиции последующих
    const originalLength = error.position.end - error.position.start;
    const delta = suggestion.length - originalLength;
    setErrors(prevErrors => {
        if (!prevErrors || prevErrors.length === 0) return [];
        return prevErrors
            .filter(e =>
                !(e.position.start === error.position.start && e.position.end === error.position.end)
            )
            .map(e => {
                if (e.position.start >= error.position.end) {
                    // Сдвигаем все ошибки, идущие после заменённого слова
                    return {
                        ...e,
                        position: {
                            start: e.position.start + delta,
                            end: e.position.end + delta
                        }
                    };
                }
                return e;
            });
    });

    console.log('[SpellChecker] Text replaced:', {
        originalLength: text.length,
        newLength: newText.length,
        replacedAt: error.position.start,
        timestamp: new Date().toISOString(),
    });

    // Фокус на textarea после изменения
    setTimeout(() => {
        const newCursorPos = error.position.start + suggestion.length;
        textareaRef.current?.setSelectionRange(newCursorPos, newCursorPos);
        textareaRef.current?.focus();
    }, 0);
};

export const highlightText = ({
    text,
    errors,
    selectedError
}: HighlightTextProps): React.ReactNode[] => {
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
                className={`bg-red-50 dark:bg-red-950/30 border-b-2 border-red-500 transition-all duration-200 relative ${isSelected
                        ? 'bg-red-200 dark:bg-red-900/50 border-red-700 dark:border-red-500 shadow-md shadow-red-500/30'
                        : 'hover:bg-red-100 dark:hover:bg-red-900/40'
                    }`}
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

export const handleTextareaClick = ({
    errors,
    text,
    textareaRef,
    setSelectedError
}: HandleTextareaClickProps) => {
    if (!textareaRef.current || errors.length === 0) return;

    // Используем setTimeout, чтобы получить позицию курсора после клика
    setTimeout(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;

        const cursorPos = textarea.selectionStart;

        // Находим ошибку, которая содержит позицию курсора
        // Проверяем, что слово в ошибке соответствует тексту в этой позиции
        const clickedError = errors.find(err => {
            const isInRange = cursorPos >= err.position.start && cursorPos < err.position.end;
            if (!isInRange) return false;

            // Дополнительная проверка: слово в ошибке должно соответствовать тексту
            // Используем text (текущий текст), так как пользователь кликает на него
            const actualWord = text.slice(err.position.start, err.position.end);
            const matches = actualWord === err.word || actualWord.toLowerCase() === err.word.toLowerCase();

            console.log('[SpellChecker] Checking error:', {
                errorWord: err.word,
                actualWord,
                position: err.position,
                cursorPos,
                matches,
            });

            return matches;
        });

        // Если не нашли точное совпадение, ищем ближайшую ошибку
        if (!clickedError) {
            // Находим ошибку с минимальным расстоянием до позиции курсора
            const closestError = errors.reduce((closest, err) => {
                const distance = Math.min(
                    Math.abs(cursorPos - err.position.start),
                    Math.abs(cursorPos - err.position.end)
                );
                const closestDistance = closest ? Math.min(
                    Math.abs(cursorPos - closest.position.start),
                    Math.abs(cursorPos - closest.position.end)
                ) : Infinity;

                return distance < closestDistance ? err : closest;
            }, null as SpellError | null);

            // Проверяем, что ближайшая ошибка достаточно близко (в пределах 10 символов)
            if (closestError) {
                const distance = Math.min(
                    Math.abs(cursorPos - closestError.position.start),
                    Math.abs(cursorPos - closestError.position.end)
                );

                if (distance <= 10) {
                    const actualWord = text.slice(closestError.position.start, closestError.position.end);
                    console.log('[SpellChecker] Using closest error:', {
                        errorWord: closestError.word,
                        actualWord,
                        position: closestError.position,
                        cursorPos,
                        distance,
                    });

                    // Проверяем, что слово соответствует
                    if (actualWord === closestError.word || actualWord.toLowerCase() === closestError.word.toLowerCase()) {
                        setSelectedError(closestError);
                        textarea.setSelectionRange(closestError.position.start, closestError.position.end);
                        return;
                    }
                }
            }
        } else {
            console.log('[SpellChecker] Error word clicked:', {
                word: clickedError.word,
                position: clickedError.position,
                clickPosition: cursorPos,
                suggestionsCount: clickedError.suggestions.length,
                suggestions: clickedError.suggestions,
                timestamp: new Date().toISOString(),
            });
            setSelectedError(clickedError);
            // Выделяем слово с ошибкой
            textarea.setSelectionRange(clickedError.position.start, clickedError.position.end);
        }
    }, 10);
};

export const getErrorSuggestions = ({
    error,
    text,
    debouncedText,
    handleSuggestionClick
}: GetErrorSuggestionsProps): React.ReactElement | null => {
    if (!error || error.suggestions.length === 0) return null;

    // Получаем актуальное слово из текста, чтобы убедиться, что оно соответствует
    // Используем текущий text для отображения, но проверяем по debouncedText
    const actualWord = text.slice(error.position.start, error.position.end);
    const debouncedWord = debouncedText.slice(error.position.start, error.position.end);

    // Если слово в debouncedText не совпадает с ошибкой, значит текст изменился
    if (debouncedWord !== error.word && debouncedWord.toLowerCase() !== error.word.toLowerCase()) {
        console.warn('[SpellChecker] Word mismatch:', {
            errorWord: error.word,
            debouncedWord,
            actualWord,
            position: error.position,
        });
        return null;
    }

    // Фильтруем предложения, убирая пустые и слишком короткие
    const validSuggestions = error.suggestions.filter(s =>
        s && s.trim().length > 0 && s !== actualWord
    );

    if (validSuggestions.length === 0) return null;

    console.log('[SpellChecker] Displaying suggestions:', {
        errorWord: error.word,
        actualWord,
        position: error.position,
        suggestionsCount: validSuggestions.length,
        suggestions: validSuggestions,
        timestamp: new Date().toISOString(),
    });

    return (
        <div
            className="bg-white dark:bg-gray-800 border-2 border-indigo-500 dark:border-indigo-400 rounded-lg p-6 mt-4 shadow-lg"
            style={{ animation: 'slideDown 0.3s ease-out' }}
        >
            <div className="flex items-center gap-4 mb-4 pb-3 border-b border-gray-200 dark:border-gray-700">
                <span className="font-semibold text-red-500 dark:text-red-400 text-lg">{actualWord}</span>
                <span className="text-gray-600 dark:text-gray-400 text-sm">Предложения:</span>
            </div>
            <div className="flex flex-wrap gap-2">
                {validSuggestions.map((suggestion, index) => (
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