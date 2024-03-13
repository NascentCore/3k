import { PageContainer } from '@ant-design/pro-components';
import { Button, Space, Table } from 'antd';
import React, { useEffect, useState } from 'react';
import dayjs from 'dayjs';
import FineTuningDrawer from './FineTuningDrawer';
import InferenceDrawer from './InferenceDrawer';
import { apiResourceDatasets, apiResourceModels } from '@/services';
import { formatFileSize } from '@/utils';

const Admin: React.FC = () => {
  const [resourceModels, setResourceModels] = useState([]);
  useEffect(() => {
    apiResourceModels({}).then((res) => {
      setResourceModels(res);
    });
  }, []);
  return (
    <PageContainer>
      <Table
        columns={[
          {
            title: '模型名称',
            dataIndex: 'id',
            key: 'id',
            align: 'center',
            width: 150,
          },
          {
            title: '所有者',
            dataIndex: 'owner',
            key: 'owner',
            align: 'center',
            width: 150,
          },
          {
            title: '模型体积',
            dataIndex: 'size',
            key: 'size',
            align: 'center',
            width: 150,
            render: (text) => {
              return formatFileSize(text);
            },
          },
          {
            title: '操作',
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Space>
                  {record?.tag?.includes('finetune') && <FineTuningDrawer record={record} />}
                  {record?.tag?.includes('inference') && <InferenceDrawer record={record} />}
                </Space>
              </>
            ),
          },
        ]}
        dataSource={resourceModels}
        scroll={{ y: 'calc(100vh - 300px)' }}
      />
    </PageContainer>
  );
};

export default Admin;
