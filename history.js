// 全局变量
let allHistoryData = [];
let filteredHistoryData = [];
let currentUser = null;

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检查认证
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/login.html';
        return;
    }

    // 获取用户信息
    currentUser = {
        username: localStorage.getItem('userName'),
        token: token
    };

    // 加载历史记录
    loadHistoryData();
});

// 加载历史记录数据
async function loadHistoryData() {
    try {
        const response = await fetch('/api/chat/history', {
            headers: {
                'Authorization': `Bearer ${currentUser.token}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            allHistoryData = data;
            filteredHistoryData = [...data];
            
            // 更新统计信息
            updateStats();
            
            // 渲染历史记录
            renderHistory();
        } else {
            showError('加载历史记录失败');
        }
    } catch (error) {
        console.error('Error loading history:', error);
        showError('网络错误，请稍后重试');
    }
}

// 更新统计信息
function updateStats() {
    const totalMessages = allHistoryData.length * 2; // 每条记录包含用户和助手消息
    const totalConversations = allHistoryData.length;
    
    // 计算首次咨询时间
    let firstMessage = '-';
    if (allHistoryData.length > 0) {
        const firstRecord = allHistoryData[0];
        if (firstRecord.timestamp) {
            const date = new Date(firstRecord.timestamp);
            firstMessage = date.toLocaleDateString('zh-CN');
        }
    }

    document.getElementById('totalMessages').textContent = totalMessages;
    document.getElementById('totalConversations').textContent = totalConversations;
    document.getElementById('firstMessage').textContent = firstMessage;
}

// 渲染历史记录
function renderHistory() {
    const historyList = document.getElementById('historyList');
    
    if (filteredHistoryData.length === 0) {
        historyList.innerHTML = `
            <div class="no-history">
                <div>📝</div>
                <h3>暂无历史记录</h3>
                <p>您还没有任何聊天记录，开始您的第一次医疗咨询吧！</p>
            </div>
        `;
        return;
    }

    let html = '';
    
    // 按时间倒序排列
    const sortedData = [...filteredHistoryData].sort((a, b) => {
        const dateA = new Date(a.timestamp || 0);
        const dateB = new Date(b.timestamp || 0);
        return dateB - dateA;
    });

    sortedData.forEach((record, index) => {
        const timestamp = new Date(record.timestamp || Date.now());
        const formattedDate = timestamp.toLocaleString('zh-CN');
        
        html += `
            <div class="history-item">
                <div class="history-meta">
                    <span>对话 #${sortedData.length - index}</span>
                    <span>${formattedDate}</span>
                </div>
                <div class="history-content">
                    <div class="message-pair user-message">
                        <div class="message-label">用户</div>
                        <div class="message-text">${escapeHtml(record.user || record.userMessage || '')}</div>
                    </div>
                    <div class="message-pair assistant-message">
                        <div class="message-label">医疗助手</div>
                        <div class="message-text">${escapeHtml(record.assistant || record.assistantMessage || '')}</div>
                    </div>
                </div>
            </div>
        `;
    });

    historyList.innerHTML = html;
}

// 筛选历史记录
function filterHistory() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;

    filteredHistoryData = allHistoryData.filter(record => {
        // 搜索筛选
        if (searchTerm) {
            const userMessage = (record.user || record.userMessage || '').toLowerCase();
            const assistantMessage = (record.assistant || record.assistantMessage || '').toLowerCase();
            if (!userMessage.includes(searchTerm) && !assistantMessage.includes(searchTerm)) {
                return false;
            }
        }

        // 日期筛选
        if (dateFrom || dateTo) {
            const recordDate = new Date(record.timestamp || Date.now());
            const recordDateStr = recordDate.toISOString().split('T')[0];
            
            if (dateFrom && recordDateStr < dateFrom) {
                return false;
            }
            if (dateTo && recordDateStr > dateTo) {
                return false;
            }
        }

        return true;
    });

    renderHistory();
}

// 清除筛选条件
function clearFilters() {
    document.getElementById('searchInput').value = '';
    document.getElementById('dateFrom').value = '';
    document.getElementById('dateTo').value = '';
    
    filteredHistoryData = [...allHistoryData];
    renderHistory();
}

// 导出历史记录
function exportHistory() {
    if (filteredHistoryData.length === 0) {
        alert('没有可导出的记录');
        return;
    }

    // 准备导出数据
    const exportData = filteredHistoryData.map(record => ({
        时间: new Date(record.timestamp || Date.now()).toLocaleString('zh-CN'),
        用户消息: record.user || record.userMessage || '',
        助手回复: record.assistant || record.assistantMessage || ''
    }));

    // 转换为CSV格式
    const headers = ['时间', '用户消息', '助手回复'];
    const csvContent = [
        headers.join(','),
        ...exportData.map(row => [
            `"${row.时间}"`,
            `"${row.用户消息.replace(/"/g, '""')}"`,
            `"${row.助手回复.replace(/"/g, '""')}"`
        ].join(','))
    ].join('\n');

    // 创建下载链接
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `医疗咨询记录_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 返回聊天页面
function goToChat() {
    window.location.href = '/chat.html';
}

// 退出登录
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('userName');
    localStorage.removeItem('userId');
    window.location.href = '/login.html';
}

// 显示错误信息
function showError(message) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = `
        <div class="no-history">
            <div>❌</div>
            <h3>加载失败</h3>
            <p>${message}</p>
            <button class="btn btn-primary" onclick="loadHistoryData()">重试</button>
        </div>
    `;
}

// HTML转义函数
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 搜索框实时筛选
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    let searchTimeout;
    
    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            filterHistory();
        }, 300);
    });
}); 