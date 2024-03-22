import { PageContainer } from '@ant-design/pro-components';
import { Space, Table } from 'antd';
import React from 'react';
import FineTuningDrawer from './FineTuningDrawer';
import InferenceDrawer from './InferenceDrawer';
import { useApiResourceModels } from '@/services';
import { formatFileSize } from '@/utils';
import { useIntl } from '@umijs/max';

const Admin: React.FC = () => {
  const intl = useIntl();
  const { data: resourceModels, mutate, isLoading }: any = useApiResourceModels();

  return (
    <PageContainer>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.modelRepository.table.column.id',
              // defaultMessage: '模型名称',
            }),
            dataIndex: 'id',
            key: 'id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.modelRepository.table.column.owner',
              // defaultMessage: '所有者',
            }),
            dataIndex: 'owner',
            key: 'owner',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.modelRepository.table.column.size',
              // defaultMessage: '模型体积',
            }),
            dataIndex: 'size',
            key: 'size',
            align: 'center',
            width: 150,
            render: (text) => {
              return formatFileSize(text);
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.modelRepository.table.column.action',
              // defaultMessage: '操作',
            }),
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
        dataSource={resourceModels || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 300px)' }}
      />
    </PageContainer>
  );
};

export default Admin;
