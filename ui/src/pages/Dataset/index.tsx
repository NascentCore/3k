import { PageContainer } from '@ant-design/pro-components';
import { Space, Table, Tabs } from 'antd';
import React from 'react';
import { useApiResourceDatasets, useApiResourceModels } from '@/services';
import { formatFileSize } from '@/utils';
import { useIntl } from '@umijs/max';

const TabTable = ({ dataSource, loading }: any) => {
  const intl = useIntl();
  return (
    <Table
      columns={[
        {
          title: intl.formatMessage({
            id: 'xxx',
            defaultMessage: '数据集名称',
          }),
          dataIndex: 'name',
          key: 'name',
          align: 'center',
          width: 150,
        },
        {
          title: intl.formatMessage({
            id: 'xxx',
            defaultMessage: '数据集大小',
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
            id: 'xxx',
            defaultMessage: '数据集说明',
          }),
          dataIndex: 'owner',
          key: 'owner',
          align: 'center',
          width: 150,
        },
        {
          title: intl.formatMessage({
            id: 'xxx',
            defaultMessage: '操作',
          }),
          dataIndex: 'action',
          key: 'action',
          width: 200,
          align: 'center',
          render: (_, record) => <></>,
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
  const { data, mutate, isLoading }: any = useApiResourceDatasets();

  const items = [
    {
      key: '1',
      label: '公共数据集',
      children: (
        <>
          <TabTable dataSource={data?.public_list || []} loading={isLoading} />
        </>
      ),
    },
    {
      key: '2',
      label: '用户数据集',
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
