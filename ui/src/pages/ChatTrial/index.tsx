/**
 * 算想云推理试用页面的 chat-ui
 * iframe 打开页面 需要传递参数 model
 * 例如: /chat-h5-model?model=google/gemma-2b-it
 * chat 功能请求的url地址是 {host}/api/v1/chat/completions
 */
import { ConfigProvider, message } from 'antd';
import React, { useEffect } from 'react';
import styles from './index.less';
import MessageInput from './MessageInput';
import ChatContainer from './ChatContainer';
import { getParameterByName, scrollH5ChatBodyToBottom } from './utils';

const Admin: React.FC = () => {
  useEffect(() => {
    window.document.title = '产品咨询';
    scrollH5ChatBodyToBottom(false);
  }, []);

  useEffect(() => {
    const model = getParameterByName('model');
    if (!model) {
      message.error('缺少参数');
      return;
    }
    sessionStorage.setItem('chat-h5-model-model', decodeURI(model));
  }, []);

  return (
    <>
      <ConfigProvider
        theme={{
          token: {
            colorPrimary: '#5a47e5',
            colorLink: '#5a47e5',
          },
        }}
      >
        <div className={styles.container}>
          <ChatContainer />
          <MessageInput />
        </div>
      </ConfigProvider>
    </>
  );
};

export default Admin;
