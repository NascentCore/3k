import React from 'react';
import RobotMessageItem from './RobotMessageItem';
import UserMessageItem from './UserMessageItem';
import { IChatItemMsg } from '@/models/chat-h5-model';

interface IProps {
  messageItem: IChatItemMsg;
}

const Index: React.FC<IProps> = ({ messageItem }) => {
  const MessageComponent = messageItem.role === 'user' ? UserMessageItem : RobotMessageItem;

  return <MessageComponent messageItem={messageItem} />;
};

export default Index;
