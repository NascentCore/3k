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
    const uploadButton = document.getElementById('upload-button');
    const clearButton = document.getElementById('clear-button');
    const imageUpload = document.getElementById('image-upload');
    const storageKey = `messageHistory_${window.location.href}`;
    let messageHistory = JSON.parse(localStorage.getItem(storageKey)) || [];
    let imageUrl = '';

    // 显示初始图片
    const initialImageElement = document.createElement('img');
    initialImageElement.src = 'assets/chatui.png';
    initialImageElement.alt = 'chatui';
    initialImageElement.style.width = '400px';
    initialImageElement.style.display = 'block';
    initialImageElement.style.margin = '0 auto 20px';
    chatMessages.appendChild(initialImageElement);

    // 显示欢迎气泡
    addMessage("您好，我是您的小助手，请问有什么可帮您。", 'assistant-message', 'assets/assistant-avatar.png');

    // 加载历史记录
    messageHistory.forEach(msg => {
        let content = '';

        if (Array.isArray(msg.content)) {
            // 如果 content 是数组，则逐个处理内容项
            msg.content.forEach((item, index) => {
                if (item.type === 'text' && item.text) {
                    content += `<p>${item.text}</p>`;
                } else if (item.type === 'image_url' && item.image_url.url) {
                    const imgId = `image-${index}-${Date.now()}`;
                    content += `<img src="${item.image_url.url}" id="${imgId}" style="max-width: 200px; margin-top: 5px; cursor: pointer;">`;
                    setTimeout(() => {
                        const imgElement = document.getElementById(imgId);
                        if (imgElement) {
                            imgElement.addEventListener('click', () => {
                                showModal(item.image_url.url);
                            });
                        }
                    }, 0);
                }
            });
        } else {
            content = msg.role === 'assistant' ? marked.parse(msg.content) : msg.content;
        }

        addMessage(content, msg.role === 'user' ? 'user-message' : 'assistant-message', msg.role === 'user' ? 'assets/user-avatar.png' : 'assets/assistant-avatar.png', msg.imageUrl);
    });

    // 滚动到屏幕底部
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 0);

    uploadButton.addEventListener('click', () => {
        imageUpload.click();
    });

    clearButton.addEventListener('click', () => {
        if (confirm('确定要清除所有消息吗？')) {
            localStorage.removeItem(storageKey);
            chatMessages.innerHTML = '';
            messageHistory = [];
            // 重新显示初始图片和欢迎信息
            chatMessages.appendChild(initialImageElement);
            addMessage("您好，我是您的小助手，请问有什么可帮您。", 'assistant-message', 'assets/assistant-avatar.png');
        }
    });

    imageUpload.addEventListener('change', async () => {
        const file = imageUpload.files[0];
        if (file) {
            imageUrl = await uploadImage(file);
            if (imageUrl) {
                showImagePreview(imageUrl);
            }
        }
    });

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

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

    // 设置 textarea 的初样式
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

    async function sendMessage() {
        const message = userInput.value.trim();

        if (message || imageUrl) {
            addMessage(message, 'user-message', 'assets/user-avatar.png', imageUrl);
            userInput.value = '';
            autoResizeTextarea();

            if (imageUrl) {
                const content = [];
                content.push({ type: 'text', text: message });
                content.push({ type: 'image_url', image_url: { url: imageUrl }});
                messageHistory.push({ role: 'user', content });
            } else {
                messageHistory.push({ role: 'user', content: message });
            }

            localStorage.setItem(storageKey, JSON.stringify(messageHistory));

            // 只发送最后10条消息
            const messagesToSend = messageHistory.slice(-10);

            // 清空并隐藏预览框
            const previewContainer = document.getElementById('preview-container');
            if (previewContainer) {
                previewContainer.innerHTML = '';
                previewContainer.style.display = 'none';
            }

            await fetchResponse(messagesToSend);
            imageUrl = ''; // 重置 imageUrl
        }
    }

    function addMessage(content, className, avatar, imageUrl = '') {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message-container', className);

        const avatarElement = document.createElement('img');
        avatarElement.src = avatar;
        avatarElement.classList.add('avatar');

        const contentContainer = document.createElement('div');
        contentContainer.style.display = 'flex';
        contentContainer.style.flexDirection = 'column';
        contentContainer.style.alignItems = 'flex-start';

        if (imageUrl) {
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;
            imgElement.style.maxWidth = '100px';
            imgElement.style.marginBottom = '5px';
            imgElement.style.cursor = 'pointer';
            imgElement.addEventListener('click', () => {
                showModal(imageUrl);
            });
            contentContainer.appendChild(imgElement);
        }

        const textElement = document.createElement('div');
        textElement.classList.add('message');
        textElement.innerHTML = className === 'assistant-message' ? marked.parse(content) : content;
        // textElement.textContent = content;
        contentContainer.appendChild(textElement);

        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.alignItems = 'flex-start';

        if (className === 'user-message') {
            wrapper.appendChild(contentContainer);
            wrapper.appendChild(avatarElement);
        } else {
            wrapper.appendChild(avatarElement);
            wrapper.appendChild(contentContainer);
        }

        messageElement.appendChild(wrapper);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function fetchResponse(messagesToSend) {
        const assistantMessage = document.createElement('div');
        assistantMessage.classList.add('message-container', 'assistant-message');

        const avatarElement = document.createElement('img');
        avatarElement.src = 'assets/assistant-avatar.png';
        avatarElement.classList.add('avatar');

        const textElement = document.createElement('div');
        textElement.classList.add('message');
        textElement.textContent = '正在思考...';

        assistantMessage.appendChild(avatarElement);
        assistantMessage.appendChild(textElement);
        chatMessages.appendChild(assistantMessage);

        try {
            const response = await fetch(`${window.location.href}/completions_turbo`, {
                method: 'POST',
                headers: {
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer xxx'
                },
                body: JSON.stringify({
                    model: '/mnt/models',
                    messages: messagesToSend.map(msg => ({
                        role: msg.role,
                        content: msg.content
                    })),
                    temperature: 1,
                    top_k: 10,
                    stream: true
                })
            });

            if (response.status === 422) {
                throw new Error('该模型不支持图片');
            }

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonString = line.slice(6).trim();
                        if (jsonString && jsonString !== '[DONE]') {
                            try {
                                const data = JSON.parse(jsonString);
                                if (data.choices && data.choices.length > 0) {
                                    const messageContent = data.choices[0].delta?.content;
                                    if (messageContent) {
                                        fullResponse += messageContent;
                                        const toolCalls = data.choices[0].message?.tool_calls || [];
                                        updateAssistantMessage(textElement, fullResponse, toolCalls);
                                    }
                                }
                            } catch (e) {
                                console.error('解析JSON时出错:', e);
                            }
                        }
                    }
                }
            }

            // 保存助手的回复到历史记录
            messageHistory.push({ role: 'assistant', content: fullResponse });
            localStorage.setItem(storageKey, JSON.stringify(messageHistory));

        } catch (error) {
            console.error('Error in fetchResponse:', error);
            textElement.textContent = '抱歉，发生了错误：' + error.message;

            // 如果是 422 错误，从历史记录中删除最后一条
            if (error.message === '该模型不支持图片') {
                messageHistory.pop();
            }
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

    // 初始调整和窗口大小变化调整
    adjustLayout();
    window.addEventListener('resize', adjustLayout);

    function showImagePreview(url) {
        let previewContainer = document.getElementById('preview-container');
        
        if (!previewContainer) {
            previewContainer = document.createElement('div');
            previewContainer.id = 'preview-container';
            document.body.appendChild(previewContainer);
        }

        // 清空之前的预览
        // previewContainer.innerHTML = '';
        previewContainer.style.position = 'fixed';
        previewContainer.style.bottom = '80px'; // 固定在上传按钮上方
        previewContainer.style.right = 'calc(50% - 380px)'; // 根据布局调整位置
        previewContainer.style.display = 'flex';
        previewContainer.style.flexWrap = 'wrap';
        previewContainer.style.justifyContent = 'flex-end';
        previewContainer.style.marginBottom = '10px';
        previewContainer.style.backgroundColor = '#333';
        previewContainer.style.padding = '10px';
        previewContainer.style.borderRadius = '8px';

        const imgPreview = document.createElement('img');
        imgPreview.src = url;
        imgPreview.style.maxWidth = '100px';
        imgPreview.style.marginRight = '10px';
        imgPreview.style.cursor = 'pointer';
        previewContainer.appendChild(imgPreview);

        imgPreview.addEventListener('click', () => {
            showModal(url);
        });
    }

    function showModal(url) {
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

        const modalImage = document.createElement('img');
        modalImage.src = url;
        modalImage.style.maxWidth = '90%';
        modalImage.style.maxHeight = '90%';
        modalImage.style.borderRadius = '10px';

        modal.appendChild(modalImage);
        document.body.appendChild(modal);

        modal.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(`${window.location.href}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('上传图片失败');
            }

            const data = await response.json();

            if (data.code === 'image_repeated') {
                return data.images; // 返回重复图片的 URL
            }

            return `${window.location.href}/${data.file_path}`; // 返回新上传图片的 URL
        } catch (error) {
            console.error('图片上传错误:', error);
            return '';
        }
    }
});
