import React, { useEffect, useState } from 'react';
import { Button, Drawer, Popconfirm, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import BuildingImage from './BuildingImage';
import { apiDeleteJobJupyterlab, apiGetJobJupyterlab, useApiGetJobJupyterlab } from '@/services';
import { formatFileSize, getToken } from '@/utils';

const Index: React.FC = ({ tableDataSourceRes, mutate, isLoading }: any) => {
  const intl = useIntl();
  const [buildingImageOpen, setBuildingImageOpen] = React.useState(false);
  const [buildingImageRecord, setBuildingImageRecord] = React.useState({});

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.instance_name',
              defaultMessage: '实例名称',
            }),
            dataIndex: 'instance_name',
            key: 'instance_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.cpu_count',
              defaultMessage: 'CPU',
            }),
            dataIndex: 'cpu_count',
            key: 'cpu_count',
            align: 'center',
            width: 80,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.memory',
              defaultMessage: 'MEM',
            }),
            dataIndex: 'memory',
            key: 'memory',
            align: 'center',
            width: 100,
            render: (text) => {
              return formatFileSize(text);
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.gpu_product',
              defaultMessage: 'GPU',
            }),
            dataIndex: 'gpu_product',
            key: 'gpu_product',
            align: 'center',
            width: 200,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.status',
              defaultMessage: '状态',
            }),
            dataIndex: 'status',
            key: 'status',
            align: 'center',
            width: 100,
          },
          {
            title: intl.formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.table.column.action',
              defaultMessage: '操作',
            }),
            dataIndex: 'action',
            key: 'action',
            align: 'center', 
            render: (_, record) => (
              <>
                <Space>
                  <Button
                    type={'link'}
                    onClick={() => {
                      window.open(record?.url);
                    }}
                  >
                    JupyterLab
                  </Button>
                  <Button
                    type={'link'}
                    onClick={() => {
                      window.open(record?.url);
                    }}
                  >
                    LLaMA-Factory
                  </Button>
                  <Button
                    type={'link'}
                    onClick={() => {
                      setBuildingImageOpen(true);
                      setBuildingImageRecord(record);
                    }}
                  >
                    {intl.formatMessage({
                      id: 'pages.jupyterLab.JupyterLabTab.table.column.action.buildBtn',
                      defaultMessage: '构建镜像',
                    })}
                  </Button>
                  <Popconfirm
                    title={intl.formatMessage({ id: 'pages.global.confirm.title' })}
                    description={intl.formatMessage({
                      id: 'pages.global.confirm.delete.description',
                    })}
                    onConfirm={() => {
                      apiDeleteJobJupyterlab({ data: { job_name: record?.job_name } }).then(() => {
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
          id: 'pages.jupyterLab.JupyterLabTab.table.column.action.buildBtn',
          defaultMessage: '构建镜像',
        })}
        placement="right"
        onClose={() => setBuildingImageOpen(false)}
        open={buildingImageOpen}
      >
        <BuildingImage
          record={buildingImageRecord}
          onChange={() => setBuildingImageOpen(false)}
          onCancel={() => setBuildingImageOpen(false)}
        />
      </Drawer>
    </>
  );
};

export default Index;
