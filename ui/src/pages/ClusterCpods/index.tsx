import { PageContainer } from '@ant-design/pro-components';
import { Card, Space, Table, Tabs } from 'antd';
import React from 'react';
import { useApiClusterCpods } from '@/services';
import { formatFileSize } from '@/utils';
import { useIntl } from '@umijs/max';

const Index: React.FC = () => {
  const intl = useIntl();
  const { data, mutate, isLoading }: any = useApiClusterCpods();
  console.log(1111, { data });
  const dataKeys = Object.keys(data);
  return (
    <PageContainer>
      {dataKeys.map((title) => (
        <>
          <Card title={<>集群id: {title}</>}>
            <Table
              columns={[
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'cpod_id',
                  }),
                  dataIndex: 'cpod_id',
                  key: 'cpod_id',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'cpod_version',
                  }),
                  dataIndex: 'cpod_version',
                  key: 'cpod_version',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'gpu_allocatable',
                  }),
                  dataIndex: 'gpu_allocatable',
                  key: 'gpu_allocatable',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'gpu内存',
                  }),
                  dataIndex: 'gpu_mem',
                  key: 'gpu_mem',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'gpu类型',
                  }),
                  dataIndex: 'gpu_prod',
                  key: 'gpu_prod',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'gpu_total',
                  }),
                  dataIndex: 'gpu数量',
                  key: 'gpu_total',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: 'gpu版本',
                  }),
                  dataIndex: 'gpu_vendor',
                  key: 'gpu_vendor',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: '创建时间',
                  }),
                  dataIndex: 'create_time',
                  key: 'create_time',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'xxx',
                    defaultMessage: '更新时间',
                  }),
                  dataIndex: 'update_time',
                  key: 'update_time',
                  align: 'center',
                  width: 150,
                },
              ]}
              dataSource={data[title] || []}
              loading={isLoading}
              scroll={{ y: 'calc(100vh - 350px)' }}
            />
          </Card>
        </>
      ))}
    </PageContainer>
  );
};

export default Index;
