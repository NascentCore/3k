import React, { useState } from 'react';
import { Button, Modal, Image } from 'antd';
import { useIntl } from '@umijs/max';
import demoImage from './assets/GPUpic.png';

const App: React.FC = () => {
  const intl = useIntl();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const showModal = () => {
    setIsModalOpen(true);
  };

  const handleCancel = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <Button type="link" onClick={showModal}>
        {intl.formatMessage({
          id: 'pages.userJob.table.column.action.detail',
          // defaultMessage: '详情',
        })}
      </Button>
      <Modal
        title={intl.formatMessage({
          id: 'pages.userJob.table.column.action.detail',
          // defaultMessage: '详情',
        })}
        open={isModalOpen}
        footer={null}
        width={700}
        onCancel={handleCancel}
      >
        <Image width={'100%'} src={demoImage} />
      </Modal>
    </>
  );
};

export default App;
