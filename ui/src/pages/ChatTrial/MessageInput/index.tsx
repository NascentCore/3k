import { ExclamationCircleFilled } from '@ant-design/icons';
import { useModel } from '@umijs/max';
import { Modal } from 'antd';
import React, { useState } from 'react';
import styles from './index.less';
import { getCommonSettingConfigured } from '../utils/commonSettingConfigured';
import { SubmitKey } from '../utils/interface';
const { confirm } = Modal;

const shouldSubmit = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
  const submitKey = getCommonSettingConfigured()?.submitKey;
  // Fix Chinese input method "Enter" on Safari
  if (e.keyCode == 229) return false;
  if (e.key !== 'Enter') return false;
  if (e.key === 'Enter' && e.nativeEvent.isComposing) return false;
  return (
    (submitKey === SubmitKey.AltEnter && e.altKey) ||
    (submitKey === SubmitKey.CtrlEnter && e.ctrlKey) ||
    (submitKey === SubmitKey.ShiftEnter && e.shiftKey) ||
    (submitKey === SubmitKey.MetaEnter && e.metaKey) ||
    (submitKey === SubmitKey.Enter && !e.altKey && !e.ctrlKey && !e.shiftKey && !e.metaKey)
  );
};
const Index: React.FC = () => {
  const { chatStore, activeChat, deleteChatStore, questionAction } = useModel('chat-h5-model');
  const chatState = chatStore[activeChat]?.state;
  const [userMsgValue, setUserMsgValue] = useState('');

  const sendMsg = async () => {
    if (chatState === 'loading' || userMsgValue.trim() === '') {
      return;
    }
    questionAction({ chatid: activeChat, userMsgValue });
    setUserMsgValue('');
  };

  const deleteMsg = () => {
    Modal.confirm({
      title: '确认',
      content: '确认清除所有消息？',
      okText: '确认',
      okType: 'danger',
      cancelText: '取消',
      onOk() {
        deleteChatStore(activeChat);
      },
      onCancel() {},
    });
  };
  return (
    <>
      <div className={styles.MessageInputInner}>
        <input
          className={styles.sendInput}
          placeholder={'请在这里输入您的问题'}
          value={userMsgValue}
          onInput={(e: any) => {
            setUserMsgValue(e.target.value);
          }}
          onKeyDown={(event: any) => {
            if (shouldSubmit(event)) {
              sendMsg();
              event.preventDefault();
            }
          }}
        />
        <div className={styles.deleteBtn} onClick={deleteMsg}></div>
        <div className={styles.sendBtn} onClick={sendMsg}></div>
      </div>
    </>
  );
};

export default Index;
