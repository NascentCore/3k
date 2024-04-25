import React from 'react';
import { Button, Popconfirm, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import { apiDeleteJobJupyterImage, useApiGetJobJupyterImageversion } from '@/services';
import { copyTextToClipboard } from '@/utils';

interface IProps {
  record?: any;
}

const Index: React.FC = ({ record }: IProps) => {
  const intl = useIntl();

  const {
    data: dataSourceRes,
    mutate,
    isLoading,
  } = useApiGetJobJupyterImageversion({
    params: {
      instance_name: record?.image_name,
    },
  });

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.image_name',
              defaultMessage: '镜像名称',
            }),
            dataIndex: 'image_name',
            key: 'image_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.tag',
              defaultMessage: 'tag',
            }),
            dataIndex: 'tag_name',
            key: 'tag_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.image_size',
              defaultMessage: '镜像大小',
            }),
            dataIndex: 'image_size',
            key: 'image_size',
            align: 'center',
            width: 100,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.push_time',
              defaultMessage: '推送时间',
            }),
            dataIndex: 'push_time',
            key: 'push_time',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action',
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
                      console.log('复制镜像地址');
                      copyTextToClipboard(record?.full_name);
                    }}
                  >
                    {intl.formatMessage({
                      id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action.copy',
                      defaultMessage: '复制镜像地址',
                    })}
                  </Button>
                  <Popconfirm
                    title={intl.formatMessage({ id: 'pages.global.confirm.title' })}
                    description={intl.formatMessage({
                      id: 'pages.global.confirm.delete.description',
                    })}
                    onConfirm={() => {
                      apiDeleteJobJupyterImage({
                        data: { image_name: record.image_name, tag_name: record.tag_name },
                      }).then(() => {
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
        dataSource={dataSourceRes?.data || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Index;
