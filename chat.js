document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const userName = document.getElementById('userName');

    // Check authentication
    const token = localStorage.getItem('token');
    const storedUserName = localStorage.getItem('userName');
    const userId = localStorage.getItem('userId') || generateUserId();
    
    if (!token) {
        window.location.href = '/index.html';
        return;
    }

    function generateUserId() {
        const id = 'user_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('userId', id);
        return id;
    }

    userName.textContent = storedUserName || 'User';

    // Logout handler
    logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('token');
        localStorage.removeItem('userName');
        localStorage.removeItem('userId');
        window.location.href = '/index.html';
    });

    // Send message handler
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');
        messageInput.value = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ 
                    message,
                    user_id: userId
                })
            });

            if (response.ok) {
                const data = await response.json();
                addMessage(data.response, 'assistant');
            } else {
                addMessage('抱歉，我遇到了一些问题。请稍后再试。', 'system');
            }
        } catch (error) {
            console.error('Chat error:', error);
            addMessage('抱歉，我遇到了一些问题。请稍后再试。', 'system');
        }
    }

    // Add message to chat with improved formatting
    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const formattedText = text.replace(/\n/g, '<br>');
        messageDiv.innerHTML = formattedText;
        
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();
        messageDiv.appendChild(timestamp);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Load chat history
    async function loadChatHistory() {
        try {
            const response = await fetch('/api/chat/history', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const history = await response.json();
                history.forEach(msg => {
                    addMessage(msg.message, msg.type);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    loadChatHistory();
}); 