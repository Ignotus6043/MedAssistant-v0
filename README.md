# Medical Assistant Website

A web-based medical assistant platform that provides an interactive chat interface for users to communicate with an AI-powered medical assistant. The platform includes both client and admin interfaces for managing users and monitoring conversations.

## Features

### Client Features
- User registration and login
- Interactive chat interface with the medical assistant
- Message history
- Secure authentication
- Responsive design

### Admin Features
- Dashboard with real-time statistics
- User management (view, edit, delete users)
- Chat history monitoring
- API key management
- Filtering and search capabilities

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- DeepSeek API key (for the AI assistant)

## Installation

1. Clone the repository or download the files to your local machine.

2. Navigate to the project directory:
```bash
cd "website test"
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project directory with the following content:
```
SECRET_KEY=your-secret-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```
Replace `your-secret-key-here` with a secure random string and `your-deepseek-api-key-here` with your DeepSeek API key.

## Running the Application

1. Start the Flask backend server:
```bash
python app.py
```
The backend will run on http://localhost:5000

2. In a new terminal, start the frontend server:
```bash
python -m http.server 8000
```
The frontend will be available at http://localhost:8000

## Usage

### Client Access

1. Open your browser and navigate to http://localhost:8000
2. You'll see the login/registration page
3. To create a new account:
   - Click the "Register" tab
   - Fill in your name, email, and password
   - Click "Register"
4. To login:
   - Enter your email and password
   - Click "Login"
5. After logging in, you'll be redirected to the chat interface
6. Type your medical questions in the chat box and press Enter or click "Send"
7. The AI assistant will respond to your queries

### Admin Access

1. Open your browser and navigate to http://localhost:8000/admin-login.html
2. Use the following credentials to login:
   - Email: admin@medicalassistant.com
   - Password: admin123
3. After logging in, you'll be redirected to the admin dashboard
4. The admin interface provides several sections:
   - Dashboard: View statistics about users and conversations
   - Users: Manage user accounts
   - Chat History: Monitor conversations
   - Settings: Configure the DeepSeek API key

## Data Storage

- User information is stored persistently in `users.json`
- Chat history is stored in-memory and will be cleared when the server restarts
- Admin credentials are hardcoded in the application (change these in production)

## API Endpoints

### Client Endpoints
- POST `/api/register` - Register a new user
- POST `/api/login` - User login
- POST `/api/chat` - Send a message to the AI assistant
- GET `/api/chat/history` - Get chat history

### Admin Endpoints
- POST `/api/admin/login` - Admin login
- GET `/api/admin/stats` - Get dashboard statistics
- GET `/api/admin/users` - Get user list
- GET `/api/admin/chats` - Get chat history
- POST `/api/admin/settings` - Update settings

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Protected API endpoints
- CORS support
- Secure password storage

## Development Notes

This is a basic implementation. For production use, consider:
1. Adding a proper database (e.g., PostgreSQL)
2. Implementing proper error handling
3. Adding input validation
4. Adding rate limiting
5. Implementing proper session management
6. Adding HTTPS
7. Adding proper logging
8. Adding user roles and permissions
9. Adding data backup and recovery
10. Adding proper testing

## Troubleshooting

1. If you can't connect to the backend:
   - Ensure the Flask server is running on port 5000
   - Check if the `.env` file is properly configured

2. If the AI assistant isn't responding:
   - Verify your DeepSeek API key in the admin settings
   - Check the backend console for error messages

3. If you can't login:
   - Ensure you're using the correct email and password
   - Check if the registration was successful

4. If admin login fails:
   - Verify you're using the correct admin credentials
   - Check if the backend server is running

## License

This project is licensed under the MIT License - see the LICENSE file for details. 