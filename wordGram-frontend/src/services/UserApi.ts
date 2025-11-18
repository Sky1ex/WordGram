import type { createUserDTO, UserDTO } from "@/DTO/userDTO";

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export class UserApi {
    private baseUrl: string;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    public async createUser(user: createUserDTO) {
        try {
            const response = await fetch(`${this.baseUrl}/api/users`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(user),
            });

            localStorage.setItem('email', user.email);
            localStorage.setItem('password', user.password)

            if (!response.ok) {
                console.error('[UserApi] HTTP error:', {
                    status: response.status,
                    statusText: response.statusText,
                    timestamp: new Date().toISOString(),
                });
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            window.location.reload();
        } catch (error) {
            console.error('[UserApi] Error checking spelling:', {
                error,
                timestamp: new Date().toISOString(),
            });
            throw error;
        }
    }

    public async LogIn({ email, password }: { email: string, password: string }) {
        try {
            const response = await fetch(`${this.baseUrl}/api/users/LogIn`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email,
                    password
                }),
            });

            if (!response.ok) {
                console.error('[UserApi] HTTP error:', {
                    status: response.status,
                    statusText: response.statusText,
                    timestamp: new Date().toISOString(),
                });
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const user: UserDTO = await response.json();

            localStorage.setItem('id', user.id.toString());
            localStorage.setItem('email', user.email);
            localStorage.setItem('password', password);

            window.location.reload();
        } catch (error) {
            console.error('[UserApi] Error during login:', {
                error,
                timestamp: new Date().toISOString(),
            });
            throw error;
        }
    }

    public async LogOut() {
        localStorage.clear();
    }

}