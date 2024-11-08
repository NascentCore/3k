function detectDevice() {
    if (window.innerWidth > 768) {
        return "Desktop";
    } else {
        return "Mobile";
    }
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

    // 创建一个隐藏的 div 来测量文本
    const hiddenDiv = document.createElement('div');
    hiddenDiv.classList.add('hidden-div');
    hiddenDiv.style.position = 'absolute';
    hiddenDiv.style.top = '-9999px';
    hiddenDiv.style.left = '-9999px';
    hiddenDiv.style.visibility = 'hidden';
    hiddenDiv.style.whiteSpace = 'pre-wrap';
    hiddenDiv.style.wordWrap = 'break-word';
    document.body.appendChild(hiddenDiv);

    // 设置 textarea 的初始样式
    const initialHeight = 25; // 设置一个较小的初始高度
    userInput.style.height = initialHeight + 'px';
    userInput.style.overflowY = 'hidden';

    // 动态调整 textarea 高度
    userInput.addEventListener('input', autoResizeTextarea);

    function autoResizeTextarea() {
        hiddenDiv.innerHTML = userInput.value.replace(/\n/g, '<br>') + '<br>';
        hiddenDiv.style.width = userInput.offsetWidth + 'px';
        
        const contentHeight = hiddenDiv.offsetHeight;
        
        if (contentHeight > initialHeight) {
            userInput.style.height = contentHeight + 'px';
        } else {
            userInput.style.height = initialHeight + 'px';
        }
    }

    // 显示图片
    const imageElement = document.createElement('img');
    imageElement.src = '21book.png';
    imageElement.alt = '21book';
    imageElement.style.width = '200px'; // 调整图片大小
    imageElement.style.display = 'block';
    imageElement.style.margin = '0 auto 20px'; // 居中并底部间距
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

    // 显示录音状态
    document.getElementById('voice-button').addEventListener('click', function() {
        const recordingIndicator = document.getElementById('recording-indicator');
        // 显示录音状态
        recordingIndicator.style.display = 'inline';

        // 检查浏览器是否支持SpeechRecognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error('浏览器不支持SpeechRecognition');
            recordingIndicator.style.display = 'none';
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'zh-CN'; // 设置语言为中文
        recognition.interimResults = false; // 只获取最终结果
        recognition.maxAlternatives = 1;

        recognition.start();

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript; // 将识别结果填入输入框
            autoResizeTextarea(); // 调整高度
            recordingIndicator.style.display = 'none'; // 隐藏录音状态
        };

        recognition.onerror = (event) => {
            console.error('语音识别错误:', event.error);
            recordingIndicator.style.display = 'none'; // 隐藏录音状态
        };

        recognition.onend = () => {
            recordingIndicator.style.display = 'none'; // 确保在识别结束时隐藏录音状态
        };
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, 'user-message', 'user-avatar.png');
            userInput.value = '';
            autoResizeTextarea(); // 重置高度
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

            let fullChunk = '';
            let fullResponse = '';
            let sourceDocuments = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                fullChunk += chunk;
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
            filterFull(fullChunk);
        } catch (error) {
            console.error('Error in fetchResponse:', error);
            textElement.textContent = '抱歉，发生了错误：' + error.message;
        }

        function filterFull(chunk){
            let fullResponse = '';
            let sourceDocuments = [];
            console.log('response.text()',  chunk);
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.replace('data: ',''));
                        if (data.response) {
                            fullResponse = data.response;
                        }
                        if (data.source_documents) {
                            sourceDocuments = sourceDocuments.concat(data.source_documents);
                        } else if (data.retrieval_documents) {
                            sourceDocuments = sourceDocuments.concat(data.retrieval_documents);
                        }
                        debouncedUpdateAssistantMessage(textElement, fullResponse, sourceDocuments);
                    } catch (e) {
                        console.error('解析JSON时出错:', e);
                        console.log(line);
                    }
                }
            }
    
            console.log('fullResponse', fullResponse);
            updateAssistantMessage(textElement, fullResponse, sourceDocuments);
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
        modal.style.alignItems = 'flex-start';
        modal.style.zIndex = '1000';
        modal.style.overflowY = 'auto';
        modal.style.paddingTop = '60px'; // 为返回按钮留出空间

        const modalContent = document.createElement('div');
        modalContent.style.backgroundColor = '#333';
        modalContent.style.color = '#fff';
        modalContent.style.padding = '20px';
        modalContent.style.borderRadius = '10px';
        modalContent.style.width = '90%';
        modalContent.style.maxWidth = '760px'; // 与输入框最大宽度一致
        modalContent.style.marginBottom = '20px'; // 底部留出一些空间

        const backButton = document.createElement('button');
        backButton.textContent = '返回';
        backButton.style.position = 'fixed';
        backButton.style.top = '10px';
        backButton.style.left = '10px';
        backButton.style.padding = '5px 10px';
        backButton.style.backgroundColor = '#007bff';
        backButton.style.color = 'white';
        backButton.style.border = 'none';
        backButton.style.borderRadius = '5px';
        backButton.style.cursor = 'pointer';
        backButton.style.zIndex = '1001'; // 确保按钮始终在最上层

        const decodedContent = content.replace(/\\n/g, '\n').replace(/\\'/g, "'");

        modalContent.innerHTML = `<h2 style="margin-top: 0;">${fileName}</h2>${marked.parse(decodedContent)}`;

        modal.appendChild(backButton);
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // 点击模态框外部区域关闭
        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                document.body.removeChild(modal);
            }
        });

        // 点击返回按钮关闭
        backButton.onclick = () => document.body.removeChild(modal);

        // 阻止模态内容的点击事件冒泡到模态框
        modalContent.addEventListener('click', function(event) {
            event.stopPropagation();
        });
    };

    // 根据设备类型调整布局
    function adjustLayout() {
        const deviceType = detectDevice();
        const chatMessages = document.getElementById('chat-messages');
        const inputContainer = document.getElementById('input-container');

        if (deviceType === "Desktop") {
            chatMessages.style.maxWidth = inputContainer.style.maxWidth;
        } else {
            chatMessages.style.maxWidth = '100%';
        }
    }

    // 初始调整和窗口大小变化时调整
    adjustLayout();
    window.addEventListener('resize', adjustLayout);
});
