export interface UserDTO {
    id: string;
    username: string;
    email: string;
    firstName: string;
    lastName: string;
    createdAt: Date;
    isActive: boolean;
}

export interface createUserDTO {
    username: string;
    email: string;
    password: string;
}