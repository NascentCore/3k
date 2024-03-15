import React, { useState } from 'react';
import { Button, Modal, Image } from 'antd';
import demoImage from './assets/GPUpic.png';

const App: React.FC = () => {
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
        详情
      </Button>
      <Modal title="详情" open={isModalOpen} footer={null} width={700} onCancel={handleCancel}>
        <Image width={'100%'} src={demoImage} />
      </Modal>
    </>
  );
};

export default App;
