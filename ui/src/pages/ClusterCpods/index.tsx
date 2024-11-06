import { PageContainer } from '@ant-design/pro-components';
import { Card, Table } from 'antd';
import React from 'react';
import { useApiClusterCpods } from '@/services';
import { useIntl } from '@umijs/max';

const Index: React.FC = () => {
  const intl = useIntl();
  const { data, isLoading }: any = useApiClusterCpods();
  console.log(1111, { data });
  const dataKeys = Object.keys(data || {});
  return (
    <PageContainer>
      {dataKeys.map((title) => (
        <>
          <Card
            title={
              <>
                {intl.formatMessage({
                  id: 'pages.ClusterCpods.card.title',
                  defaultMessage: '集群ID',
                })}
                : {title}
              </>
            }
            style={{ marginBottom: 15 }}
          >
            <Table
              columns={[
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.node_name',
                    defaultMessage: '节点名称',
                  }),
                  dataIndex: 'node_name',
                  key: 'node_name',
                  align: 'center',
                  width: 150,
                },
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.node_type',
                    defaultMessage: '节点类型',
                  }),
                  key: 'node_type',
                  align: 'center',
                  width: 150,
                  render: () => 'control-plane,worker',
                },
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.gpu_prod',
                    defaultMessage: 'GPU型号',
                  }),
                  dataIndex: 'gpu_prod',
                  key: 'gpu_prod',
                  align: 'center',
                  width: 200,
                },
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.gpu_mem',
                    defaultMessage: 'GPU内存',
                  }),
                  dataIndex: 'gpu_mem',
                  key: 'gpu_mem',
                  align: 'center',
                  width: 100,
                  render: (bytes) => {
                    const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                    return `${gb} GB`;
                  },
                },
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.gpu_total',
                    defaultMessage: 'GPU总数',
                  }),
                  dataIndex: 'gpu_total',
                  key: 'gpu_total',
                  align: 'center',
                  width: 100,
                },
                {
                  title: intl.formatMessage({
                    id: 'pages.ClusterCpods.table.column.gpu_allocatable',
                    defaultMessage: '可分配GPU',
                  }),
                  dataIndex: 'gpu_allocatable',
                  key: 'gpu_allocatable',
                  align: 'center',
                  width: 100,
                },
                // {
                //   title: intl.formatMessage({
                //     id: 'pages.ClusterCpods.table.column.cpod_version',
                //     defaultMessage: 'CPod版本',
                //   }),
                //   dataIndex: 'cpod_version',
                //   key: 'cpod_version',
                //   align: 'center',
                //   width: 150,
                // },

                // {
                //   title: intl.formatMessage({
                //     id: 'pages.ClusterCpods.table.column.gpu_vendor',
                //     defaultMessage: 'GPU厂商',
                //   }),
                //   dataIndex: 'gpu_vendor',
                //   key: 'gpu_vendor',
                //   align: 'center',
                //   width: 150,
                // },
                // {
                //   title: intl.formatMessage({
                //     id: 'pages.ClusterCpods.table.column.create_time',
                //     defaultMessage: '创建时间',
                //   }),
                //   dataIndex: 'create_time',
                //   key: 'create_time',
                //   align: 'center',
                //   width: 150,
                // },
                // {
                //   title: intl.formatMessage({
                //     id: 'pages.ClusterCpods.table.column.update_time',
                //     defaultMessage: '更新时间',
                //   }),
                //   dataIndex: 'update_time',
                //   key: 'update_time',
                //   align: 'center',
                //   width: 150,
                // },
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
