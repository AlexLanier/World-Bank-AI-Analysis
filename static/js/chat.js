/**
 * Chat UI component for AI chatbot
 */
class ChatBot {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.messages = [];
        this.isOpen = false;
        this.isTyping = false;
        this.init();
    }

    init() {
        if (!this.container) return;
        
        this.createUI();
        this.attachEventListeners();
    }

    createUI() {
        this.container.innerHTML = `
            <div class="chat-header">
                <h3>🤖 AI Assistant</h3>
                <button class="chat-toggle" id="chatToggle">−</button>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="chat-welcome">
                    <p>👋 Hello! I'm your AI assistant for the World Bank Loan Prediction Dashboard.</p>
                    <p>I can help you with:</p>
                    <ul>
                        <li>Understanding loan predictions</li>
                        <li>Answering questions about the data</li>
                        <li>Explaining model features</li>
                    </ul>
                    <p>Try asking: "How does the model work?" or "What data is available?"</p>
                </div>
            </div>
            <div class="chat-input-container">
                <div class="chat-typing" id="chatTyping" style="display: none;">
                    <span>AI is thinking...</span>
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
                <div class="chat-input-wrapper">
                    <input type="text" id="chatInput" placeholder="Ask me anything..." />
                    <button id="chatSend">Send</button>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        const toggleBtn = document.getElementById('chatToggle');
        const sendBtn = document.getElementById('chatSend');
        const input = document.getElementById('chatInput');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (input) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
    }

    toggle() {
        this.isOpen = !this.isOpen;
        const chatBody = this.container.querySelector('.chat-body');
        if (chatBody) {
            chatBody.style.display = this.isOpen ? 'flex' : 'none';
        }
        const toggleBtn = document.getElementById('chatToggle');
        if (toggleBtn) {
            toggleBtn.textContent = this.isOpen ? '−' : '+';
        }
    }

    async sendMessage() {
        const input = document.getElementById('chatInput');
        if (!input) return;

        const message = input.value.trim();
        if (!message || this.isTyping) return;

        // Add user message to UI
        this.addMessage('user', message);
        input.value = '';

        // Show typing indicator
        this.showTyping();

        try {
            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    history: this.messages
                })
            });

            const data = await response.json();

            this.hideTyping();

            if (data.success) {
                this.addMessage('assistant', data.response);
                // Update conversation history
                this.messages.push({ role: 'user', content: message });
                this.messages.push({ role: 'assistant', content: data.response });
            } else {
                this.addMessage('assistant', `Error: ${data.error || 'Could not get response'}`);
            }
        } catch (error) {
            this.hideTyping();
            this.addMessage('assistant', `Error: ${error.message}`);
            console.error('Chat error:', error);
        }
    }

    addMessage(role, content) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message chat-message-${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'chat-message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTyping() {
        this.isTyping = true;
        const typingIndicator = document.getElementById('chatTyping');
        if (typingIndicator) {
            typingIndicator.style.display = 'flex';
        }
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    hideTyping() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('chatTyping');
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }
    }
}

// Initialize chat when DOM is ready
function initChat() {
    if (document.getElementById('chatBot')) {
        window.chatBot = new ChatBot('chatBot');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChat);
} else {
    // DOM already loaded
    initChat();
}

