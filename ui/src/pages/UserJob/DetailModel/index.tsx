import React, { useState } from 'react';
import { Button, Modal, Image } from 'antd';
import { useIntl } from '@umijs/max';
import demoImage from './assets/GPUpic.png';

const App: React.FC = ({ record }: any) => {
  const intl = useIntl();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const webUrl = `http://grafana.llm.sxwl.ai:30003/d/a85faaa0-8ff6-4f11-ac27-24cbb0fa4ee9/job-detail?orgId=1&var-ns=${record.userId}&var-pod=${record.jobName}`;
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
        width={1200}
        onCancel={handleCancel}
      >
        <iframe src={webUrl} style={{ width: '100%', height: 500, border: 'none' }}></iframe>
      </Modal>
    </>
  );
};

export default App;
