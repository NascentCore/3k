import { PageContainer } from '@ant-design/pro-components';
import { Space, Table, Tabs } from 'antd';
import React from 'react';
import FineTuningDrawer from './FineTuningDrawer';
import InferenceDrawer from './InferenceDrawer';
import { useApiResourceModels } from '@/services';
import { formatFileSize, removeUserIdPrefixFromPath } from '@/utils';
import { useIntl } from '@umijs/max';

const TabTable = ({ dataSource, loading }: any) => {
  console.log('dataSource', dataSource);
  const intl = useIntl();
  return (
    <Table
      columns={[
        {
          title: intl.formatMessage({
            id: 'pages.modelRepository.table.column.id',
            // defaultMessage: '模型名称',
          }),
          dataIndex: 'name',
          key: 'name',
          align: 'center',
          width: 150,
          render: (_) => removeUserIdPrefixFromPath(_),
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
      dataSource={dataSource}
      loading={loading}
      scroll={{ y: 'calc(100vh - 350px)' }}
    />
  );
};

const Index: React.FC = () => {
  const intl = useIntl();
  const { data, isLoading }: any = useApiResourceModels();

  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.modelRepository.tab.title.public',
        defaultMessage: '公共模型',
      }),
      children: (
        <>
          <TabTable dataSource={data?.public_list || []} loading={isLoading} />
        </>
      ),
    },
    {
      key: '2',
      label: intl.formatMessage({
        id: 'pages.modelRepository.tab.title.user',
        defaultMessage: '私有模型',
      }),
      children: (
        <>
          <TabTable dataSource={data?.user_list || []} loading={isLoading} />
        </>
      ),
    },
  ];
  return (
    <PageContainer>
      <Tabs defaultActiveKey="1" items={items} />
    </PageContainer>
  );
};

export default Index;
