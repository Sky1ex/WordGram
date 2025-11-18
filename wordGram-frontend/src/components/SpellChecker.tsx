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
import type { UserDTO, createUserDTO } from '../DTO/userDTO';
import SignInDialog from './SignInDialog';
import InputCheck from './Inputcheck';
import { UserApi } from '@/services/UserApi';

export default function SpellChecker() {
	const [user, setUser] = useState<createUserDTO | null>(null);

	const userApi = new UserApi();

	useEffect(() => {
		try {
			const email = localStorage.getItem('email');
			const password = localStorage.getItem('password');
			if (email && password) setUser({ username: '', email: email, password: password });
		}
		catch {
			console.log('пользователь не найден');
		}
	}, [])

	return (
		<div className="max-w-7xl mx-auto p-8 md:p-4 font-sans ">
			<Card className='w-full p-4 flex flex-row justify-between bg-gray-100'>
				<div className='place-self-center text-2xl'>Word Gram</div>
				<DropdownMenu>
					<DropdownMenuTrigger asChild>
						{user ? (
							<div className='flex flex-row gap-2 cursor-pointer hover:bg-gray-200 rounded-md p-2'>
								<Avatar className=''>
									<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" width={40} className='rounded-3xl' />
									<AvatarFallback>CN</AvatarFallback>
								</Avatar>
								<div className='place-self-center'>{user.email}</div>
							</div>
						) :
							(
								<div className='flex flex-row gap-2 cursor-pointer hover:bg-gray-200 rounded-md p-2'>
									<Avatar className=''>
										<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" width={40} className='rounded-3xl' />
										<AvatarFallback>CN</AvatarFallback>
									</Avatar>
									<div className='place-self-center'>Вход не выполнен</div>
								</div>
							)}

					</DropdownMenuTrigger>
					<DropdownMenuContent className="w-56" align="start">
						<DropdownMenuLabel>Аккаунт</DropdownMenuLabel>
						<DropdownMenuGroup>
							{user ? (
								<DropdownMenuItem>
									<User className='size-4' />
									Профиль
								</DropdownMenuItem>
							) : (
								<SignInDialog />
							)}
							<DropdownMenuItem>
								<Settings className='size-4' />
								Найстройки
							</DropdownMenuItem>
							<DropdownMenuItem onClick={userApi.LogOut}>
								<LogOut className='size-4' />
								Выйти
							</DropdownMenuItem>
						</DropdownMenuGroup>
					</DropdownMenuContent>
				</DropdownMenu>
			</Card>
			<InputCheck />
		</div>
	);
}
