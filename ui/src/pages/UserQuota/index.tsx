import { PageContainer } from '@ant-design/pro-components';
import { Button, Popconfirm, Space, Table, message } from 'antd';
import React from 'react';
import { useGetApiQuota } from '@/services';
import { useIntl } from '@umijs/max';
import EditDrawer from './EditDrawer';

const Admin: React.FC = () => {
  const intl = useIntl();
  const { data: quotaRes, mutate, isLoading }: any = useGetApiQuota();

  return (
    <PageContainer>
      <div style={{ marginBottom: 20, textAlign: 'right' }}>
        <EditDrawer type="add" />
      </div>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.userQuota.table.column.name',
              defaultMessage: '用户',
            }),
            dataIndex: 'user_id',
            key: 'user_id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userQuota.table.column.role',
              defaultMessage: '资源类型',
            }),
            dataIndex: 'resource',
            key: 'resource',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userQuota.table.column.gpu_product',
              defaultMessage: '资源配额',
            }),
            dataIndex: 'quota',
            key: 'quota',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userQuota.table.column.action',
              defaultMessage: '操作',
            }),
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Space>
                  <EditDrawer record={record} type="edit" onChange={() => mutate()} />

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
                </Space>
              </>
            ),
          },
        ]}
        dataSource={quotaRes?.data || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 300px)' }}
      />
    </PageContainer>
  );
};

export default Admin;
