import React, { useState } from 'react';
import { Button, Drawer, Popconfirm, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import ImageDetail from './ImageDetail';
import { apiDeleteJobJupyterImage } from '@/services';
import moment from 'moment';

const Index: React.FC = ({ tableDataSourceRes, mutate, isLoading }: any) => {
  const intl = useIntl();
  const [detailDrawerOpen, setDetailDrawerOpen] = useState(false);
  const [detailRecord, setDetailRecord] = useState<any>(void 0);

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.table.image_name',
              defaultMessage: '镜像名称',
            }),
            dataIndex: 'image_name',
            key: 'image_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.table.create_time',
              defaultMessage: '创建时间',
            }),
            dataIndex: 'created_at',
            key: 'created_at',
            align: 'center',
            width: 150,
            render: (text: string) => {
              return moment(text).format('YYYY-MM-DD HH:mm:ss');
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.table.push_time',
              defaultMessage: '更新时间',
            }),
            dataIndex: 'updated_at',
            key: 'updated_at',
            align: 'center',
            width: 150,
            render: (text: string) => {
              return moment(text).format('YYYY-MM-DD HH:mm:ss');
            },
          },

          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.table.action',
              defaultMessage: '操作',
            }),
            dataIndex: 'action',
            key: 'action',
            align: 'center',
            width: 150,
            render: (_, record) => (
              <>
                <Space>
                  <Button
                    type={'link'}
                    onClick={() => {
                      setDetailRecord(record);
                      setDetailDrawerOpen(true);
                    }}
                  >
                    {intl.formatMessage({
                      id: 'pages.jupyterLab.ImageManagementTab.table.action.detail',
                      defaultMessage: '详情',
                    })}
                  </Button>
                  <Popconfirm
                    title={intl.formatMessage({ id: 'pages.global.confirm.title' })}
                    description={intl.formatMessage({
                      id: 'pages.global.confirm.delete.description',
                    })}
                    onConfirm={() => {
                      apiDeleteJobJupyterImage({ data: { ...record } }).then(() => {
                        mutate();
                      });
                    }}
                    okText={intl.formatMessage({ id: 'pages.global.confirm.okText' })}
                    cancelText={intl.formatMessage({ id: 'pages.global.confirm.cancelText' })}
                  >
                    <Button type="link">
                      {intl.formatMessage({ id: 'pages.global.confirm.delete.button' })}
                    </Button>
                  </Popconfirm>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={tableDataSourceRes?.data || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.jupyterLab.ImageManagementTab.table.action.detail',
          defaultMessage: '详情',
        })}
        placement="right"
        onClose={() => setDetailDrawerOpen(false)}
        open={detailDrawerOpen}
      >
        {detailDrawerOpen && <ImageDetail record={detailRecord} />}
      </Drawer>
    </>
  );
};

export default Index;
