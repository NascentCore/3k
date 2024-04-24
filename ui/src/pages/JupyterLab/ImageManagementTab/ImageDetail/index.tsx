import React, { useEffect, useState } from 'react';
import { Button, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import { apiGetJobJupyterImage } from '@/services';

interface IProps {
  record?: any;
}

const Index: React.FC = ({ record }: IProps) => {
  const intl = useIntl();

  const [dataSource, setDatasource] = useState([]);
  useEffect(() => {
    apiGetJobJupyterImage({
      data: {
        image_name: record?.image_name,
      },
    }).then((res) => {
      setDatasource(res?.data);
    });
  }, []);

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
                  <Button type={'link'}>
                    {intl.formatMessage({
                      id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action.copy',
                      defaultMessage: '复制镜像地址',
                    })}
                  </Button>
                  <Button type={'link'}>
                    {intl.formatMessage({
                      id: 'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action.delete',
                      defaultMessage: '删除',
                    })}
                  </Button>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={dataSource || []}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Index;
