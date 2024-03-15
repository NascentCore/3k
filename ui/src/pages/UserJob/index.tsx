import { apiDeleteUserJob, useApiGetUserJob } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Popconfirm, Space, Table, theme } from 'antd';
import React, { useEffect, useState } from 'react';
import DetailModel from './DetailModel';

const Welcome: React.FC = () => {
  const {
    data: userJobList,
    mutate,
    isLoading,
  } = useApiGetUserJob({
    params: {
      current: 1,
      size: 1000,
    },
  });

  return (
    <PageContainer>
      <Table
        columns={[
          {
            title: '任务名称',
            dataIndex: 'jobName',
            key: 'jobName',
            align: 'center',
            width: 150,
          },
          {
            title: 'GPU数量',
            dataIndex: 'gpuNumber',
            key: 'gpuNumber',
            align: 'center',
            width: 100,
          },
          {
            title: 'GPU型号',
            dataIndex: 'gpuType',
            key: 'gpuType',
            align: 'center',
            width: 150,
          },
          {
            title: 'CKPT 路径',
            dataIndex: 'ckptPath',
            key: 'ckptPath',
            align: 'center',
            width: 150,
          },
          {
            title: 'Model保存路径',
            dataIndex: 'modelPath',
            key: 'modelPath',
            align: 'center',
            width: 150,
          },
          // {
          //   title: '镜像名称',
          //   dataIndex: 'beanName',
          //   key: 'beanName',
          //   align: 'center',
          //   width: 150,
          // },
          // {
          //   title: '训练数据源',
          //   dataIndex: 'trainingsource',
          //   key: 'trainingsource',
          //   align: 'center',
          //   width: 150,
          // },
          // {
          //   title: '挂载路径',
          //   dataIndex: 'mountPath',
          //   key: 'mountPath',
          //   align: 'center',
          //   width: 150,
          // },
          {
            title: '任务类型',
            dataIndex: 'jobType',
            key: 'jobType',
            align: 'center',
            width: 100,
          },
          {
            title: '运行状态',
            dataIndex: 'workStatus',
            key: 'workStatus',
            align: 'center',
            width: 100,
            render: (text) => {
              if (text === 1) {
                return <>{'运行失败'}</>;
              } else if (text === 2) {
                return <>{'运行成功'}</>;
              } else {
                return <>{'运行中'}</>;
              }
            },
          },
          {
            title: '创建时间',
            dataIndex: 'createTime',
            key: 'createTime',
            align: 'center',
            width: 150,
          },
          {
            title: '操作',
            fixed: 'right',
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Space>
                  <DetailModel />
                  <Popconfirm
                    title="提示"
                    description="确认删除?"
                    onConfirm={() => {
                      apiDeleteUserJob({ data: [record?.id] }).then(() => {
                        mutate();
                      });
                    }}
                    onCancel={() => {}}
                    okText="是"
                    cancelText="否"
                  >
                    <Button type="link">删除</Button>
                  </Popconfirm>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={userJobList?.content || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </PageContainer>
  );
};

export default Welcome;
