import React, { useState } from 'react';
import { Button, Modal, Image } from 'antd';
import { useIntl } from '@umijs/max';
import demoImage from './assets/GPUpic.png';
import moment from 'moment';

const App: React.FC = ({ record }: any) => {
  const intl = useIntl();
  // const [isModalOpen, setIsModalOpen] = useState(false);

  // const showModal = () => {
  //   setIsModalOpen(true);
  // };

  const openWeb = () => {
    // 辅助函数：根据record计算from和to参数
    const calculateParams = () => {
      const from = moment(record.createTime).valueOf(); // 创建时间的时间戳
      let to;
      if (record.workStatus === 8) {
        to = moment(record.updateTime).valueOf(); // 状态为2时，使用更新时间的时间戳
      } else {
        to = 'now'; // 否则，使用当前时间的时间戳
      }
      const refresh = to === 'now' ? '&refresh=auto' : '';
      return { from, to, refresh };
    };
    // 构建webUrl
    const { from, to, refresh } = calculateParams();
    const webUrl = `http://grafana.llm.sxwl.ai:30005/d/a85faaa0-8ff6-4f11-ac27-24cbb0fa4ee9/job-detail?orgId=1&var-ns=${
      record.userId}&var-pod=${record.jobName}&from=${from}&to=${to}${refresh}`;
    window.open(webUrl);
  };

  // const handleCancel = () => {
  //   setIsModalOpen(false);
  // };

  return (
    <>
      <Button type="link" onClick={openWeb}>
        {intl.formatMessage({
          id: 'pages.userJob.table.column.action.detail',
          // defaultMessage: '详情',
        })}
      </Button>
      {/* <Modal
        title={intl.formatMessage({
          id: 'pages.userJob.table.column.action.detail',
          // defaultMessage: '详情',
        })}
        open={isModalOpen}
        footer={null}
        width={1200}
        onCancel={handleCancel}
      ></Modal> */}
    </>
  );
};

export default App;
