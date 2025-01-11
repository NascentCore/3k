import {
  cahtActionH5_ModalType,
  generateUUID,
  scrollH5ChatBodyToBottom,
} from '@/pages/ChatTrial/utils';
import storage from '@/pages/ChatTrial/utils/store';
import { useEffect, useState } from 'react';

export const chat_store_key = 'Chat_Store_H5_Model';

export interface IChatItemMsg {
  content: string;
  role: 'user' | 'assistant';
  id: string;
  date: string;
  source_documents?: any[];
  show_images?: any[];
  raw?: any;
}
export interface IChatItem {
  id: string;
  state: 'success' | 'loading';
  messages: IChatItemMsg[];
}

export interface IChatStore {
  [key: string]: IChatItem;
}

export interface IKnowledgeListItem {
  kb_id: string;
  kb_name: string;
}

const testChatStore: IChatStore = {
  demo: {
    id: 'demo',
    state: 'success',
    messages: [],
  },
};

export default () => {
  // 管理 chat 数据
  const [chatModel, setChatModel] = useState<any>('');
  const [chatStore, setChatStore] = useState<IChatStore>(testChatStore);
  const [activeChat, setActiveChat] = useState('demo');
  console.log('chatStore', chatStore);
  useEffect(() => {
    const init = async () => {
      const _chatStoreJson: any = await storage.getItem(chat_store_key);
      if (_chatStoreJson) {
        setChatStore(_chatStoreJson);
        setActiveChat(Object.keys(_chatStoreJson)[0]);
      }
    };
    init();
  }, []);

  useEffect(() => {
    // storage.setItem(chat_store_key, chatStore);
  }, [chatStore]);

  // 新增聊天数据
  const addChatMsg = (id: string, msg: IChatItemMsg) => {
    setChatStore((chatStore: any) => {
      const _chatStore = JSON.parse(JSON.stringify(chatStore));
      _chatStore[id].messages.push(msg);
      setTimeout(() => {
        scrollH5ChatBodyToBottom(id, true);
      }, 100);
      return _chatStore;
    });
  };

  /**
   * 更新聊天数据
   * @param id chatid
   * @param chatMsgItem
   */
  const updateChatMsg = (id: string, chatMsgItem: IChatItemMsg) => {
    setChatStore((chatStore: any) => {
      const _chatStore = JSON.parse(JSON.stringify(chatStore));
      const chatItem = _chatStore[id];
      const index = chatItem.messages.findIndex((x: any) => x.id === chatMsgItem.id);
      chatItem.messages[index] = chatMsgItem;
      return _chatStore;
    });
    setTimeout(() => {
      scrollH5ChatBodyToBottom(id, true);
    }, 100);
  };

  /**
   * 设置聊天状态
   * @param id
   * @param state
   */
  const setChatItemState = (id: string, state: string) => {
    setChatStore((chatStore: any) => {
      const _chatStore = JSON.parse(JSON.stringify(chatStore));
      _chatStore[id].state = state;
      return _chatStore;
    });
  };

  /**
   * 清空聊天
   */
  const deleteChatStore = (id: string) => {
    setChatStore((chatStore: any) => {
      const _chatStore = JSON.parse(JSON.stringify(chatStore));
      _chatStore[id].messages = [];
      return _chatStore;
    });
  };

  const questionAction = ({ chatid, userMsgValue }: { chatid: string; userMsgValue: string }) => {
    const chatMsgItemUser: IChatItemMsg = {
      content: userMsgValue,
      role: 'user',
      date: '',
      id: generateUUID(),
    };
    addChatMsg(chatid, chatMsgItemUser);
    setChatItemState(chatid, 'loading');
    const msgId = generateUUID();
    const chatMsgItem: IChatItemMsg = {
      content: '正在搜索...',
      role: 'assistant',
      id: msgId,
      date: '',
    };
    addChatMsg(chatid, chatMsgItem);
    cahtActionH5_ModalType({
      id: msgId,
      question: userMsgValue,
      onMessage: (chatMsgItem: IChatItemMsg) => {
        updateChatMsg(chatid, chatMsgItem);
      },
      onSuccess: (chatMsgItem: IChatItemMsg) => {
        updateChatMsg(chatid, chatMsgItem);
        setChatItemState(chatid, 'success');
      },
    });
  };

  const reQuestionAction = (chatMsgItem: IChatItemMsg) => {
    console.log('reQuestionAction', chatMsgItem);
    for (const key in chatStore) {
      for (let index = 0; index < chatStore[key].messages.length; index++) {
        const _chatMsgItem = chatStore[key].messages[index];
        if (_chatMsgItem.id === chatMsgItem.id) {
          questionAction({ chatid: key, userMsgValue: chatStore[key].messages[index - 1].content });
          return;
        }
      }
    }
  };

  return {
    chatStore,
    addChatMsg,
    updateChatMsg,
    deleteChatStore,
    setChatItemState,
    questionAction,
    reQuestionAction,
    activeChat,
    setChatModel,
  };
};
