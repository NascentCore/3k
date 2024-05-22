import { apiDeleteUserJob, useApiGetUserJob } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Popconfirm, Space, Table, message, theme } from 'antd';
import React, { useEffect, useState } from 'react';
import DetailModel from './DetailModel';
import { useIntl } from '@umijs/max';

const Welcome: React.FC = () => {
  const intl = useIntl();
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
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.jobName',
              // defaultMessage: '任务名称',
            }),
            dataIndex: 'jobName',
            key: 'jobName',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.gpuNumber',
              // defaultMessage: 'GPU数量',
            }),
            dataIndex: 'gpuNumber',
            key: 'gpuNumber',
            align: 'center',
            width: 100,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.gpuType',
              // defaultMessage: 'GPU型号',
            }),
            dataIndex: 'gpuType',
            key: 'gpuType',
            align: 'center',
            width: 150,
          },
          // {
          //   title: 'CKPT 路径',
          //   dataIndex: 'ckptPath',
          //   key: 'ckptPath',
          //   align: 'center',
          //   width: 150,
          // },
          // {
          //   title: 'Model保存路径',
          //   dataIndex: 'modelPath',
          //   key: 'modelPath',
          //   align: 'center',
          //   width: 150,
          // },
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
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.jobType',
              // defaultMessage: '任务类型',
            }),
            dataIndex: 'jobType',
            key: 'jobType',
            align: 'center',
            width: 100,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.workStatus',
              // defaultMessage: '运行状态',
            }),
            dataIndex: 'workStatus',
            key: 'workStatus',
            align: 'center',
            width: 100,
            render: (text) => {
              if (text === 1) {
                return (
                  <>
                    {intl.formatMessage({
                      id: 'pages.userJob.table.column.workStatus.status.1',
                      // defaultMessage: '运行失败',
                    })}
                  </>
                );
              } else if (text === 2) {
                return (
                  <>
                    {intl.formatMessage({
                      id: 'pages.userJob.table.column.workStatus.status.2',
                      // defaultMessage: '运行成功',
                    })}
                  </>
                );
              } else {
                return (
                  <>
                    {intl.formatMessage({
                      id: 'pages.userJob.table.column.workStatus.status.3',
                      // defaultMessage: '运行中',
                    })}
                  </>
                );
              }
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.createTime',
              // defaultMessage: '创建时间',
            }),
            dataIndex: 'createTime',
            key: 'createTime',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.userJob.table.column.action',
              // defaultMessage: '操作',
            }),
            fixed: 'right',
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Space>
                  <DetailModel record={record} />
                  <Popconfirm
                    title={intl.formatMessage({
                      id: 'pages.global.confirm.title',
                    })}
                    description={intl.formatMessage({
                      id: 'pages.global.confirm.delete.description',
                    })}
                    onConfirm={() => {
                      apiDeleteUserJob({ data: { job_id: record?.jobName } }).then(() => {
                        message.success(
                          intl.formatMessage({
                            id: 'pages.global.confirm.delete.success',
                            defaultMessage: '删除成功',
                          }),
                        );
                        mutate();
                      });
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
        dataSource={userJobList?.content || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </PageContainer>
  );
};

export default Welcome;
