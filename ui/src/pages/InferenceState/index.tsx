import { apiDeleteInference, apiGetInference, apiGetUserJob, useApiGetInference } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Popconfirm, Popover, Space, Table, Typography, message, theme } from 'antd';
import React, { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const { data: inferenceList, mutate, isLoading } = useApiGetInference();

  const statusMap: any = {
    notassigned: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.notassigned',
      defaultMessage: '未下发',
    }),
    assigned: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.assigned',
      defaultMessage: '已下发',
    }),
    datapreparing: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.datapreparing',
      defaultMessage: '数据准备中',
    }),
    pending: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.pending',
      defaultMessage: '启动中',
    }),
    paused: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.paused',
      defaultMessage: '已暂停',
    }),
    pausing: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.pausing',
      defaultMessage: '暂停中',
    }),
    running: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.running',
      defaultMessage: '运行中',
    }),
    failed: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.failed',
      defaultMessage: '失败',
    }),
    succeeded: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.succeeded',
      defaultMessage: '运行成功',
    }),
    deleted: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.stopped',
      defaultMessage: '已终止',
    }),
  };

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
            dataIndex: 'model_name',
            key: 'model_name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.inferenceState.table.column.category',
              defaultMessage: '类型',
            }),
            dataIndex: 'model_category',
            key: 'model_category',
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
            render: (text) => statusMap[text],
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
            width: 300,
            align: 'center',
            render: (_, record: API.InferenceInfo) => (
              <Space>
                {record.status === 'running' && (
                  <>
                    {record.model_category === 'chat' && (
                      <>
                        <Button
                          type="link"
                          onClick={() => {
                            window.open(record?.url);
                          }}
                        >
                          {intl.formatMessage({
                            id: 'pages.inferenceState.table.column.action.startChat',
                          })}
                        </Button>
                        <Popover
                          placement="left"
                          content={
                            <div>
                              <Typography.Text>
                                <pre style={{ maxWidth: 400 }}>{record.api}</pre>
                              </Typography.Text>
                              <div>
                                <Button
                                  onClick={() => {
                                    if (!navigator.clipboard) return;
                                    navigator.clipboard.writeText(record.api).then(() => {
                                      message.success(
                                        intl.formatMessage({
                                          id: 'pages.inferenceState.table.column.action.copy.success',
                                        }),
                                      );
                                    });
                                  }}
                                >
                                  {intl.formatMessage({
                                    id: 'pages.inferenceState.table.column.action.copy',
                                  })}
                                </Button>
                              </div>
                            </div>
                          }
                          title={intl.formatMessage({
                            id: 'pages.inferenceState.table.column.action.copyApiEndPoint',
                          })}
                          trigger="hover"
                        >
                          <Button type="link">
                            {intl.formatMessage({
                              id: 'pages.inferenceState.table.column.action.copyApiEndPoint',
                            })}
                          </Button>
                        </Popover>
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
                            }).then(() => {
                              message.success(
                                intl.formatMessage({
                                  id: 'pages.inferenceState.table.column.action.stop.success',
                                }),
                              );
                              mutate();
                            });
                          }}
                          okText={intl.formatMessage({
                            id: 'pages.global.confirm.okText',
                          })}
                          cancelText={intl.formatMessage({
                            id: 'pages.global.confirm.cancelText',
                          })}
                        >
                          <Button type="link">
                            {intl.formatMessage({
                              id: 'pages.inferenceState.table.column.action.stop',
                            })}
                          </Button>
                        </Popconfirm>
                      </>
                    )}
                  </>
                )}
                
                {record.status === 'deleted' && (
                  <Popconfirm
                    title={intl.formatMessage({
                      id: 'pages.global.confirm.title',
                    })}
                    description={intl.formatMessage({
                      id: 'pages.inferenceState.table.column.action.delete.confirm',
                    })}
                    onConfirm={() => {
                      apiDeleteInference({
                        params: {
                          service_name: record.service_name,
                        },
                      }).then(() => {
                        message.success(
                          intl.formatMessage({
                            id: 'pages.inferenceState.table.column.action.delete.success',
                          }),
                        );
                        mutate();
                      });
                    }}
                    okText={intl.formatMessage({
                      id: 'pages.global.confirm.okText',
                    })}
                    cancelText={intl.formatMessage({
                      id: 'pages.global.confirm.cancelText',
                    })}
                  >
                    <Button type="link">
                      {intl.formatMessage({
                        id: 'pages.inferenceState.table.column.action.delete',
                      })}
                    </Button>
                  </Popconfirm>
                )}
              </Space>
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
