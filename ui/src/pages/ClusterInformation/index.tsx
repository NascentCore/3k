import { PageContainer } from '@ant-design/pro-components';
import { Button, Popconfirm, Space, Table, message } from 'antd';
import React from 'react';
import { useGetApiNode } from '@/services';
import { formatFileSize } from '@/utils';
import { useIntl } from '@umijs/max';
import AddNodeDrawer from './AddNodeDrawer';

const Admin: React.FC = () => {
  const intl = useIntl();
  const { data: resourceModels, mutate, isLoading }: any = useGetApiNode();

  return (
    <PageContainer>
      <div style={{ marginBottom: 20, textAlign: 'right' }}>
        <AddNodeDrawer type={'add'} />
      </div>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.clusterInformation.table.column.name',
              defaultMessage: '节点名称',
            }),
            dataIndex: 'name',
            key: 'name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.clusterInformation.table.column.role',
              defaultMessage: '节点类型',
            }),
            dataIndex: 'role',
            key: 'role',
            align: 'center',
            width: 150,
            render: (_, record) => {
              return record.role?.join(',');
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.clusterInformation.table.column.gpu_product',
              defaultMessage: 'GPU资源',
            }),
            dataIndex: 'gpu_product',
            key: 'gpu_product',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.clusterInformation.table.column.gpu_count',
              defaultMessage: 'GPU数量',
            }),
            dataIndex: 'gpu_count',
            key: 'gpu_count',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.clusterInformation.table.column.action',
              defaultMessage: '操作',
            }),
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Popconfirm
                  title={intl.formatMessage({
                    id: 'pages.global.confirm.title',
                  })}
                  description={intl.formatMessage({
                    id: 'pages.global.confirm.delete.description',
                  })}
                  onConfirm={() => {
                    message.success(
                      intl.formatMessage({
                        id: 'pages.global.confirm.delete.success',
                        defaultMessage: '删除成功',
                      }),
                    );
                    mutate();
                  }}
                  onCancel={() => {}}
                  okText={intl.formatMessage({
                    id: 'pages.global.confirm.okText',
                  })}
                  cancelText={intl.formatMessage({
                    id: 'pages.global.confirm.cancelText',
                  })}
                >
                  <Button type="link">
                    {intl.formatMessage({
                      id: 'pages.global.confirm.delete.button',
                    })}
                  </Button>
                </Popconfirm>
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
