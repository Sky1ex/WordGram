import { LogIn } from "lucide-react"
import { Button } from "./ui/button"
import { Dialog, DialogClose, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "./ui/dialog"
import { DialogDescription } from "./ui/dialog"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { DropdownMenuItem } from "./ui/dropdown-menu"
import { useState } from "react"

import { UserApi } from '@/services/UserApi';

const SignInDialog = () => {

    const [isRegistering, setIsRegistering] = useState(true);
    const [username, setUsername] = useState<string>('');
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const userApi = new UserApi();

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);

        try {
            await userApi.createUser({username, email, password});
            // Успешная регистрация - можно закрыть диалог или показать сообщение
            setUsername('');
            setEmail('');
            setPassword('');
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Произошла ошибка при регистрации';
            setError(errorMessage);
            console.error('Ошибка регистрации:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogIn = async (e:React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);

        try {
            await userApi.LogIn({ email, password});
            // Успешная регистрация - можно закрыть диалог или показать сообщение
            setUsername('');
            setEmail('');
            setPassword('');
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Произошла ошибка при входе';
            setError(errorMessage);
            console.error('Ошибка входа:', err);
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <Dialog>
            <DialogTrigger asChild>
                <DropdownMenuItem onSelect={(e) => {
                    e.preventDefault();
                }}>
                    <LogIn className='size-4' />
                    Войти
                </DropdownMenuItem>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
                {isRegistering ? (
                    <form onSubmit={handleLogIn}>
                        <DialogHeader>
                            <DialogTitle>Вход в систему</DialogTitle>
                            <DialogDescription>
                                Введите ваш email и пароль для входа в систему.
                            </DialogDescription>
                        </DialogHeader>
                        <div className="grid gap-4 mt-2">
                            <div className="grid gap-3">
                                <Label htmlFor="username-1">Почта</Label>
                                <Input id="username-1" name="Почта" placeholder="Почта" value={email} onChange={(e) => setEmail(e.target.value)}/>
                            </div>
                            <div className="grid gap-3">
                                <Label htmlFor="password-1">Пароль</Label>
                                <Input id="password-1" name="Пароль" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)}/>
                            </div>
                            <div className="">
                                <a href="#" className="underline" onClick={() => setIsRegistering(false)}>Ещё не зарегистрированы?</a>
                            </div>
                        </div>
                        <DialogFooter className="flex flex-row justify-between mt-2">
                            <DialogClose asChild>
                                <Button variant="outline">Отмена</Button>
                            </DialogClose>
                            <Button type="submit">Войти</Button>
                        </DialogFooter>
                    </form>
                ) : (
                    <form onSubmit={handleRegister}>
                        <DialogHeader>
                            <DialogTitle>Регистрация</DialogTitle>
                            <DialogDescription>
                                Введите ваш email и пароль для входа в систему.
                            </DialogDescription>
                        </DialogHeader>
                        <div className="grid gap-2 mt-2">
                            <Label htmlFor="username-1">Имя</Label>
                            <Input id="username-1" name="Имя" placeholder="Имя" value={username} onChange={(e) => setUsername(e.target.value)}/>
                        </div>
                        <div className="grid gap-2 mt-2">
                            <Label htmlFor="mail-1">Почта</Label>
                            <Input id="mail-1" name="Почта" placeholder="Почта" value={email} onChange={(e) => setEmail(e.target.value)}/>
                        </div>
                        <div className="grid gap-2 mt-2">
                            <Label htmlFor="password-1">Пароль</Label>
                            <Input id="password-1" name="Пароль" type="password" placeholder="Пароль" value={password} onChange={(e) => setPassword(e.target.value)}/>
                        </div>
                        {error && (
                            <div className="mt-2 p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded">
                                {error}
                            </div>
                        )}
                        <div className="mt-2">
                            <a href="#" className="underline" onClick={(e) => {e.preventDefault(); setIsRegistering(true);}}>Уже зарегистрированы?</a>
                        </div>
                        <DialogFooter className="flex flex-row justify-between mt-2">
                            <DialogClose asChild>
                                <Button variant="outline">Отмена</Button>
                            </DialogClose>
                            <Button type="submit" disabled={isLoading}>
                                {isLoading ? 'Регистрация...' : 'Зарегистрироваться'}
                            </Button>
                        </DialogFooter>
                    </form>
                )}
            </DialogContent>
        </Dialog>
    )
}

export default SignInDialog;