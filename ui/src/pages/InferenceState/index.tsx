import { apiDeleteInference, apiGetInference, apiGetUserJob, useApiGetInference } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Popconfirm, Space, Table, message, theme } from 'antd';
import React, { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const { data: inferenceList, mutate, isLoading } = useApiGetInference();
  return (
    <PageContainer>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.service_name',
              // defaultMessage: '推理服务名称',
            }),
            dataIndex: 'service_name',
            key: 'service_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.model_id',
              // defaultMessage: '模型名称',
            }),
            dataIndex: 'model_id',
            key: 'model_id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.status',
              // defaultMessage: '推理服务状态',
            }),
            dataIndex: 'status',
            key: 'status',
            align: 'center',
            width: 150,
          },

          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.url',
              // defaultMessage: 'url',
            }),
            dataIndex: 'url',
            key: 'url',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.start_time',
              // defaultMessage: '启动时间',
            }),
            dataIndex: 'start_time',
            key: 'start_time',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.end_time',
              // defaultMessage: '终止时间',
            }),
            dataIndex: 'end_time',
            key: 'end_time',
            align: 'center',
            width: 150,
          },

          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.action',
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
                  {record.status === 'deployed' && (
                    <Button
                      type={'link'}
                      onClick={() => {
                        window.open(record?.url);
                      }}
                    >
                      {intl.formatMessage({
                        id: 'pages.inferenceState.table.column.action.startChat',
                        // defaultMessage: '启动聊天',
                      })}
                    </Button>
                  )}

                  {record.status !== 'stopped' && (
                    <Popconfirm
                      title={intl.formatMessage({
                        id: 'pages.global.confirm.title',
                      })}
                      description={intl.formatMessage({
                        id: 'pages.inferenceState.table.column.action.stop.confirm',
                      })}
                      onConfirm={() => {
                        apiDeleteInference({
                          params: {
                            service_name: record.service_name,
                          },
                        }).then((res) => {
                          message.success(
                            intl.formatMessage({
                              id: 'pages.inferenceState.table.column.action.stop.success',
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
                      <Button type={'link'}>
                        {intl.formatMessage({
                          id: 'pages.inferenceState.table.column.action.stop',
                          // defaultMessage: '终止',
                        })}
                      </Button>
                    </Popconfirm>
                  )}
                </Space>
              </>
            ),
          },
        ]}
        dataSource={inferenceList?.data || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 350px)' }}
      />
    </PageContainer>
  );
};

export default Welcome;
