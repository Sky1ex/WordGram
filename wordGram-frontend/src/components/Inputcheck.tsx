import { useState, useEffect, useRef } from 'react';
import type { SpellError } from '../types/spellCheck';
import { useDebounce } from '../hooks/useDebounce';
import {
    checkSpelling,
    handleTextChange as handleTextChangeHook,
    handleSuggestionClick as handleSuggestionClickHook,
    highlightText as highlightTextHook,
    handleTextareaClick as handleTextareaClickHook,
    getErrorSuggestions as getErrorSuggestionsHook
} from './SpellCheckerHooks';

import { Button } from "@/components/ui/button"
import { Card } from './ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@radix-ui/react-avatar';
import { DropdownMenu, DropdownMenuContent, DropdownMenuGroup, DropdownMenuItem, DropdownMenuLabel, DropdownMenuShortcut, DropdownMenuTrigger } from './ui/dropdown-menu';

import { LogOut, Settings, User } from 'lucide-react';
import type { UserDTO } from '../DTO/userDTO';
import SignInDialog from './SignInDialog';

interface SpellCheckerProps {
    language?: string;
}

const InputCheck = ({ language = 'ru' }: SpellCheckerProps) => {
    const [text, setText] = useState('');
    const [errors, setErrors] = useState<SpellError[]>([]);
    const [isChecking, setIsChecking] = useState(false);
    const [selectedError, setSelectedError] = useState<SpellError | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const skipNextCheckRef = useRef(false);
    const debouncedText = useDebounce(text, 1000);

    const [user, setUser] = useState<UserDTO | null>(null);

    useEffect(() => {
        // Если только что применили исправление — пропускаем следующую проверку
        if (skipNextCheckRef.current) {
            skipNextCheckRef.current = false;
            return;
        }
        if (debouncedText.trim().length === 0) {
            setErrors([]);
            setIsChecking(false);
            return;
        }

        checkSpelling({ setIsChecking, debouncedText, language, setErrors, selectedError, setSelectedError });
    }, [debouncedText, language]);

    const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        handleTextChangeHook({ e, text, setText, setSelectedError });
    };

    const handleSuggestionClick = (error: SpellError, suggestion: string) => {
        handleSuggestionClickHook({
            error,
            suggestion,
            text,
            setText,
            setErrors,
            setSelectedError,
            textareaRef,
            markSkipNextCheck: () => { skipNextCheckRef.current = true; }
        });
    };

    const highlightText = (text: string, errors: SpellError[]): React.ReactNode[] => {
        return highlightTextHook({ text, errors, selectedError });
    };

    const handleTextareaClick = () => {
        handleTextareaClickHook({
            errors,
            text,
            textareaRef,
            setSelectedError
        });
    };

    const getErrorSuggestions = (error: SpellError | null) => {
        return getErrorSuggestionsHook({
            error,
            text,
            debouncedText,
            handleSuggestionClick
        });
    };
    return (
        <div className="max-w-7xl mx-auto p-8 md:p-4 font-sans">

            <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-6 pb-4 border-b-2 border-gray-200 dark:border-gray-700 gap-2">
                <span className=''>Введите текст для проаверки</span>
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
                        className="w-full min-h-[400px] md:min-h-[300px] p-6 text-base leading-relaxed font-inherit border-none outline-none resize-y bg-transparent text-transparent caret-gray-900 dark:caret-white z-2 relative"
                        value={text}
                        onChange={handleTextChange}
                        onClick={handleTextareaClick}
                        placeholder="Введите текст для проверки орфографии..."
                        spellCheck={false}
                    />
                    <div
                        className="absolute inset-0 p-6 text-base leading-relaxed font-inherit whitespace-pre-wrap break-words z-1 text-gray-900 dark:text-white pointer-events-none"
                        aria-hidden="true"
                    >
                        {highlightText(text, errors)}
                    </div>
                </div>
            </div>

            <Button variant="outline">Button</Button>

            {getErrorSuggestions(selectedError)}

            {errors.length > 0 && !selectedError && (
                <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-md text-center text-gray-600 dark:text-gray-400 text-sm">
                    <p>Нажмите на выделенные слова, чтобы увидеть предложения по исправлению</p>
                </div>
            )}
        </div>
    )
}

export default InputCheck;