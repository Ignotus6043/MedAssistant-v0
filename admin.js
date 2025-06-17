document.addEventListener('DOMContentLoaded', () => {
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.admin-section');
    const adminLogoutBtn = document.getElementById('adminLogoutBtn');
    const settingsForm = document.getElementById('settingsForm');
    const userSearch = document.getElementById('userSearch');
    const userFilter = document.getElementById('userFilter');
    const dateFilter = document.getElementById('dateFilter');
    const clearHistBtn = document.getElementById('clearHist');

    // Check admin authentication
    const token = localStorage.getItem('adminToken');
    if (!token) {
        window.location.href = '/admin-login.html';
        return;
    }

    // Navigation
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.dataset.section;
            navBtns.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(section).classList.add('active');
        });
    });

    // Logout handler
    adminLogoutBtn.addEventListener('click', () => {
        localStorage.removeItem('adminToken');
        window.location.href = '/admin-login.html';
    });

    // Load dashboard stats
    async function loadDashboardStats() {
        try {
            const response = await fetch('/api/admin/stats', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const stats = await response.json();
                document.getElementById('totalUsers').textContent = stats.totalUsers;
                document.getElementById('activeChats').textContent = stats.activeChats;
                document.getElementById('totalMessages').textContent = stats.totalMessages;
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    // Load users
    async function loadUsers() {
        try {
            const response = await fetch('/api/admin/users', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const users = await response.json();
                const tbody = document.getElementById('usersTableBody');
                tbody.innerHTML = '';
                
                users.forEach(user => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${user.name}</td>
                        <td>${user.email}</td>
                        <td>${user.status}</td>
                        <td>
                            <button class="btn-secondary" onclick="viewUser('${user.id}')">View</button>
                            <button class="btn-secondary" onclick="editUser('${user.id}')">Edit</button>
                            <button class="btn-secondary" onclick="deleteUser('${user.id}')">Delete</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (error) {
            console.error('Error loading users:', error);
        }
    }

    // Load chat history
    async function loadChatHistory() {
        try {
            const response = await fetch('/api/admin/chats', {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (response.ok) {
                const chats = await response.json();
                const chatHistory = document.getElementById('chatHistory');
                chatHistory.innerHTML = '';
                
                chats.forEach(chat => {
                    const chatDiv = document.createElement('div');
                    chatDiv.className = 'chat-entry';
                    chatDiv.innerHTML = `
                        <div class="chat-header">
                            <span>${chat.userName}</span>
                            <span>${new Date(chat.timestamp).toLocaleString()}</span>
                        </div>
                        <div class="chat-content">
                            <p><strong>User:</strong> ${chat.userMessage}</p>
                            <p><strong>Assistant:</strong> ${chat.assistantMessage}</p>
                        </div>
                    `;
                    chatHistory.appendChild(chatDiv);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    // Settings form submission
    settingsForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const apiKey = document.getElementById('apiKey').value;
        const model = document.getElementById('modelSelect').value;

        try {
            const response = await fetch('/api/admin/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ apiKey, model })
            });

            if (response.ok) {
                alert('Settings saved successfully!');
            } else {
                alert('Failed to save settings.');
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            alert('An error occurred while saving settings.');
        }
    });

    // Search and filter handlers
    userSearch.addEventListener('input', loadUsers);
    userFilter.addEventListener('change', loadChatHistory);
    dateFilter.addEventListener('change', loadChatHistory);

    // Clear history button handler
    if (clearHistBtn) {
        clearHistBtn.addEventListener('click', async () => {
            if (!confirm('Clear ALL chat history?')) return;
            await fetch('/api/admin/chats/clear', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            loadChatHistory();
        });
    }

    // Initial load
    loadDashboardStats();
    loadUsers();
    loadChatHistory();
}); 