import { IChatItemMsg } from '@/models/chat-h5-model';
import React from 'react';
import assistantAvatar from './../assets/assistant-avatar.png';
import styles from './../index.less';
// import SourceDocumentsList from './SourceDocumentsList';
// import ImageList from './ImageList';
import { MarkdownContent } from '../../MarkdownContent';

interface IProps {
  messageItem: IChatItemMsg;
}

const Index: React.FC<IProps> = ({ messageItem }) => {
  // console.log('RobotMessageItem messageItem', messageItem);
  const sourceDocsCount = messageItem.source_documents?.length;

  const show_images = messageItem.show_images || [];
  return (
    <>
      <div className={styles.messageItemWrap}>
        <div
          className={styles.messageItemAvatar}
          style={{ backgroundImage: `url(${assistantAvatar})` }}
        ></div>
        <div className={styles.messageItemContent} style={{ background: '#2c2c2e', color: '#fff' }}>
          <MarkdownContent content={messageItem.content} />
          {/* 引用图片 */}
          {/* {!!(show_images && show_images.length > 0) && <ImageList show_images={show_images} />} */}
          {/* 数据来源 */}
          {/* {!!(sourceDocsCount && sourceDocsCount > 0) && (
            <>
              <div>数据来源:</div>
              <SourceDocumentsList messageItem={messageItem} />
            </>
          )} */}
        </div>
      </div>
    </>
  );
};

export default Index;
