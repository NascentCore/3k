function detectDevice() {
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    if (/android/i.test(userAgent)) {
        return "Android";
    }
    if (/iPad|iPhone|iPod/.test(userAgent) && !window.MSStream) {
        return "iOS";
    }
    return "Desktop";
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // 显示图片
    const imageElement = document.createElement('img');
    imageElement.src = '21book.png';
    imageElement.alt = '21book';
    imageElement.style.width = '200px'; // 调整图片大小
    imageElement.style.display = 'block';
    imageElement.style.margin = '0 auto 20px'; // 居中并添加底部间距
    chatMessages.appendChild(imageElement);

    // 显示欢迎气泡
    addMessage("您好，这里是 21 年法律年鉴，您有相关的问题都可以在这里向我提问。", 'assistant-message', 'assistant-avatar.png');

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    document.getElementById('voice-button').addEventListener('click', () => {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'zh-CN';
        recognition.start();

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, 'user-message', 'user-avatar.png');
            userInput.value = '';
            fetchResponse(message);
        }
    }

    function addMessage(content, className, avatar) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message-container', className);

        const avatarElement = document.createElement('img');
        avatarElement.src = avatar;
        avatarElement.classList.add('avatar');

        const textElement = document.createElement('div');
        textElement.classList.add('message');
        textElement.textContent = content;

        messageElement.appendChild(avatarElement);
        messageElement.appendChild(textElement);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function fetchResponse(message) {
        const assistantMessage = document.createElement('div');
        assistantMessage.classList.add('message-container', 'assistant-message');

        const avatarElement = document.createElement('img');
        avatarElement.src = 'assistant-avatar.png';
        avatarElement.classList.add('avatar');

        const textElement = document.createElement('div');
        textElement.classList.add('message');
        textElement.textContent = '正在思考...';

        assistantMessage.appendChild(avatarElement);
        assistantMessage.appendChild(textElement);
        chatMessages.appendChild(assistantMessage);

        try {
            const response = await fetch('http://knowledge.llm.sxwl.ai:30002/api/local_doc_qa/local_doc_chat', {
                method: 'POST',
                headers: {
                    'Accept': 'text/event-stream,application/json, text/event-stream',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,mt;q=0.6,pl;q=0.5',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: "zzp",
                    kb_ids: ["KBb42ee9c8236349d49c8329dbdece3329_240625"],
                    history: [],
                    question: message,
                    streaming: true,
                    networking: false,
                    product_source: "saas",
                    rerank: false,
                    only_need_search_results: false,
                    hybrid_search: true,
                    max_token: 1024,
                    api_base: "",
                    api_key: "",
                    model: "",
                    api_context_length: 8192,
                    chunk_size: 800,
                    top_p: 1,
                    top_k: 30,
                    temperature: 0.5
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let fullResponse = '';
            let sourceDocuments = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.response) {
                                fullResponse += data.response;
                            }
                            if (data.source_documents) {
                                sourceDocuments = sourceDocuments.concat(data.source_documents);
                            } else if (data.retrieval_documents) {
                                sourceDocuments = sourceDocuments.concat(data.retrieval_documents);
                            }
                            debouncedUpdateAssistantMessage(textElement, fullResponse, sourceDocuments);
                        } catch (e) {
                            console.error('解析JSON时出错:', e);
                        }
                    }
                }
            }
            updateAssistantMessage(textElement, fullResponse, sourceDocuments);
        } catch (error) {
            console.error('Error in fetchResponse:', error);
            textElement.textContent = '抱歉，发生了错误：' + error.message;
        }
    }

    function updateAssistantMessage(messageElement, content, sourceDocuments) {
        let htmlContent = '';
        try {
            htmlContent = marked.parse(content);
        } catch (error) {
            console.error('Error in marked parsing:', error);
            htmlContent = content;
        }
        
        if (sourceDocuments && Array.isArray(sourceDocuments) && sourceDocuments.length > 0) {
            htmlContent += '<span style="display: inline-block; margin-bottom: 5px;">数据来源：</span><div class="source-links">';
            sourceDocuments.forEach((doc, index) => {
                if (doc) {
                    const fileName = doc.file_name || doc.filename || `Document ${index + 1}`;
                    const content = doc.content || doc.text || 'No content available';
                    const escapedFileName = fileName.replace(/'/g, "\\'");
                    const escapedContent = content.replace(/'/g, "\\'").replace(/\n/g, "\\n");
                    htmlContent += `<a href="#" class="source-link" onclick="showSourceDocument('${escapedFileName}', '${escapedContent}')">${index + 1}</a>`;
                }
            });
            htmlContent += '</div>';
        }

        try {
            messageElement.innerHTML = htmlContent;
        } catch (error) {
            console.error('Error setting innerHTML:', error);
        }

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    const debouncedUpdateAssistantMessage = debounce(updateAssistantMessage, 100);

    window.showSourceDocument = function(fileName, content) {
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.left = '0';
        modal.style.top = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.zIndex = '1000';

        const modalContent = document.createElement('div');
        modalContent.style.backgroundColor = '#2c2c2e';
        modalContent.style.color = '#fff';
        modalContent.style.padding = '20px';
        modalContent.style.borderRadius = '10px';
        modalContent.style.width = '90%';
        modalContent.style.maxHeight = '80%';
        modalContent.style.overflow = 'auto';

        const closeButton = document.createElement('button');
        closeButton.textContent = '关闭';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '10px';
        closeButton.style.right = '10px';
        closeButton.style.padding = '5px 10px';
        closeButton.style.backgroundColor = '#007bff';
        closeButton.style.color = 'white';
        closeButton.style.border = 'none';
        closeButton.style.borderRadius = '5px';
        closeButton.style.cursor = 'pointer';
        closeButton.onclick = () => document.body.removeChild(modal);

        const decodedContent = content.replace(/\\n/g, '\n').replace(/\\'/g, "'");

        modalContent.innerHTML = `<h2>${fileName}</h2>${marked.parse(decodedContent)}`;
        modalContent.appendChild(closeButton);

        modal.appendChild(modalContent);
        document.body.appendChild(modal);
    };
});
