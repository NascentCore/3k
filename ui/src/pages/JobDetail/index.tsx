import { PageContainer } from '@ant-design/pro-components';
import { Button, Popconfirm, Popover, Space, Table, Typography, message } from 'antd';
import React, { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';
import { 
  apiDeleteInference, 
  apiDeleteUserJob,
  useApiGetInference, 
  useApiGetUserJob,
  apiStopInference
} from '@/services';
import DetailModel from '../UserJob/DetailModel';

// 添加类型定义
type AlignType = 'left' | 'right' | 'center';

export interface TableRecord {
  sourceType: 'inference' | 'userJob';
  status: string;
  service_name?: string;
  model_category?: string;
  url?: string;
  api?: string;
  jobId: string;
  tensor_url?: string;
}

const JobDetail: React.FC = () => {
  const intl = useIntl();
  const { data: inferenceList, mutate: mutateInference } = useApiGetInference();
  const { data: userJobList, mutate: mutateUserJob } = useApiGetUserJob({
    params: {
      current: 1,
      size: 1000,
    }
  });
  const [mergedData, setMergedData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // 状态映射
  const statusMap: any = {
    notassigned: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.notassigned',
      defaultMessage: '未下发',
    }),
    assigned: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.assigned',
      defaultMessage: '已下发',
    }),
    pending: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.pending',
      defaultMessage: '启动中',
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
    stopped: intl.formatMessage({
      id: 'pages.jupyterLab.JupyterLabTab.table.column.status.stopped',
      defaultMessage: '已终止',
    }),
  };

  // 合并数据
  useEffect(() => {
    // 确保即使数据为空也能正常工作
    const inferenceData = inferenceList?.data?.map((item: any) => ({
      ...item,
      jobId: item.service_name,
      modelName: item.model_name,
      jobType: 'Inference',
      gpuModel: item.gpu_model,
      gpuCount: item.gpu_count,
      startTime: item.start_time ? new Date(item.start_time).getTime() : 0,
      endTime: item.end_time ? new Date(item.end_time).getTime() : 0,
      status: item.status,
      sortTime: new Date(item.create_time).getTime(),
      sourceType: 'inference'
    })) || [];

    const userJobData = userJobList?.content?.map((item: any) => ({
      ...item,
      jobId: item.jobName,
      modelName: '',
      jobType: item.jobType === 'Finetune' ? 'Finetune' : 'GPUJob',
      gpuModel: item.gpuType,
      gpuCount: item.gpuNumber,
      startTime: new Date(item.createTime).getTime(),
      endTime: item.workStatus === 8 && item.updateTime ? new Date(item.updateTime).getTime() : 0,
      sortTime: new Date(item.create_time).getTime(),
      status: item.status,
      sourceType: 'userJob'
    })) || [];

    const merged = [...inferenceData, ...userJobData].sort((a, b) => {
      if (a.sortTime === 0) return 1;
      if (b.sortTime === 0) return -1;
      return b.sortTime - a.sortTime;
    });

    setMergedData(merged);
    setLoading(false);
  }, [inferenceList, userJobList]);

  const handleDelete = (record: any) => {
    if (record.sourceType === 'inference') {
      return apiDeleteInference({
        params: { service_name: record.service_name },
      }).then(() => {
        message.success('删除成功');
        mutateInference();
      });
    } else {
      return apiDeleteUserJob({
        data: { job_id: record.jobId },
      }).then(() => {
        message.success('删除成功');
        mutateUserJob();
      });
    }
  };

  const handleStop = (record: any) => {
    return apiStopInference({
      params: { service_name: record.service_name },
    }).then(() => {
      message.success('终止成功');
      mutateInference();
    });
  };

  const formatTime = (timestamp: number) => {
    if (!timestamp) return '-';
    return new Date(timestamp).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

  const columns = [
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.jobId',
        defaultMessage: '任务ID',
      }),
      dataIndex: 'jobId',
      key: 'jobId',
      width: 200,
      align: 'center' as AlignType,
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.modelName',
        defaultMessage: '模型',
      }),
      dataIndex: 'modelName',
      key: 'modelName',
      width: 150,
      align: 'center' as AlignType,
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.jobType',
        defaultMessage: '类型',
      }),
      dataIndex: 'jobType',
      key: 'jobType',
      width: 100,
      align: 'center' as AlignType,
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.gpuModel',
        defaultMessage: 'GPU型号',
      }),
      dataIndex: 'gpuModel',
      key: 'gpuModel',
      width: 150,
      align: 'center' as AlignType,
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.gpuCount',
        defaultMessage: 'GPU数量',
      }),
      dataIndex: 'gpuCount',
      key: 'gpuCount',
      width: 100,
      align: 'center' as AlignType,
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.status',
        defaultMessage: '状态',
      }),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      align: 'center' as AlignType,
      render: (text: string | undefined) => {
        if (!text) return '-';
        const key = text.toLowerCase();
        return statusMap[key] || text;
      },
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.startTime',
        defaultMessage: '启动时间',
      }),
      dataIndex: 'startTime',
      key: 'startTime',
      width: 150,
      align: 'center' as AlignType,
      render: (time: number) => formatTime(time)
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.endTime',
        defaultMessage: '终止时间',
      }),
      dataIndex: 'endTime',
      key: 'endTime',
      width: 150,
      align: 'center' as AlignType,
      render: (time: number) => formatTime(time)
    },
    {
      title: intl.formatMessage({
        id: 'pages.jobDetail.table.column.action',
        defaultMessage: '操作',
      }),
      key: 'action',
      width: 200,
      fixed: 'right' as const,
      align: 'center' as AlignType,
      render: (_: unknown, record: TableRecord) => (
        <Space size={0}>
          {record.sourceType === 'inference' && (
            <>
              {record.status === 'running' && record.model_category === 'chat' && (
                <>
                  <Button
                    type="link"
                    size="small"
                    onClick={() => {
                      window.open(record.url);
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
                              if (!navigator.clipboard) {
                                return;
                              }
                              navigator.clipboard.writeText(record.api ?? '').then(() => {
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
                    <Button type="link" size="small">
                      {intl.formatMessage({
                        id: 'pages.inferenceState.table.column.action.copyApiEndPoint',
                      })}
                    </Button>
                  </Popover>
                </>
              )}
              {record.status === 'running' && record.model_category === 'embedding' && (
                <Popover
                  placement="left"
                  content={
                    <div>
                      <Typography.Text>
                        <pre style={{ maxWidth: 400 }}>{record.url}</pre>
                      </Typography.Text>
                      <div>
                        <Button
                          onClick={() => {
                            if (!navigator.clipboard) {
                              return;
                            }
                            navigator.clipboard.writeText(record.url ?? '').then(() => {
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
                  <Button type="link" size="small">
                    {intl.formatMessage({
                      id: 'pages.inferenceState.table.column.action.copyApiEndPoint',
                    })}
                  </Button>
                </Popover>
              )}
              {record.status === 'running' ? (
                <Popconfirm
                  title={intl.formatMessage({
                    id: 'pages.global.confirm.title',
                  })}
                  description={intl.formatMessage({
                    id: 'pages.global.confirm.stop.description',
                    defaultMessage: '确认要终止该任务吗？',
                  })}
                  onConfirm={() => handleStop(record)}
                  okText={intl.formatMessage({
                    id: 'pages.global.confirm.okText',
                  })}
                  cancelText={intl.formatMessage({
                    id: 'pages.global.confirm.cancelText',
                  })}
                >
                  <Button 
                    type="link" 
                    size="small"
                  >
                    {intl.formatMessage({
                      id: 'pages.global.confirm.stop.button',
                      defaultMessage: '终止',
                    })}
                  </Button>
                </Popconfirm>
              ) : (
                <Popconfirm
                  title={intl.formatMessage({
                    id: 'pages.global.confirm.title',
                  })}
                  description={intl.formatMessage({
                    id: 'pages.global.confirm.delete.description',
                  })}
                  onConfirm={() => handleDelete(record)}
                  okText={intl.formatMessage({
                    id: 'pages.global.confirm.okText',
                  })}
                  cancelText={intl.formatMessage({
                    id: 'pages.global.confirm.cancelText',
                  })}
                >
                  <Button 
                    type="link" 
                    size="small"
                  >
                    {intl.formatMessage({
                      id: 'pages.global.confirm.delete.button',
                    })}
                  </Button>
                </Popconfirm>
              )}
            </>
          )}
          
          {record.sourceType === 'userJob' && (
            <Space size={-8}>
              <Button
                type="link"
                size="small"
                onClick={() => {
                  window.open(record.tensor_url);
                }}
              >
                {intl.formatMessage({
                  id: 'pages.userJob.table.column.action.tensorboard',
                  defaultMessage: '训练指标',
                })}
              </Button>
              <DetailModel record={record} />
              <Popconfirm
                title={intl.formatMessage({
                  id: 'pages.global.confirm.title',
                })}
                description={intl.formatMessage({
                  id: 'pages.global.confirm.delete.description',
                })}
                onConfirm={() => handleDelete(record)}
                okText={intl.formatMessage({
                  id: 'pages.global.confirm.okText',
                })}
                cancelText={intl.formatMessage({
                  id: 'pages.global.confirm.cancelText',
                })}
              >
                <Button 
                  type="link" 
                  size="small"
                >
                  {intl.formatMessage({
                    id: 'pages.global.confirm.delete.button',
                  })}
                </Button>
              </Popconfirm>
            </Space>
          )}
        </Space>
      ),
    },
  ];

  // 改其他列的 align 属性
  const formattedColumns = columns.map(column => ({
    ...column,
    align: column.align as AlignType,
  }));

  return (
    <PageContainer>
      <Table
        columns={formattedColumns}
        dataSource={mergedData}
        loading={loading}
        scroll={{ 
          y: 'calc(100vh - 350px)',
          x: 1500
        }}
      />
    </PageContainer>
  );
};

export default JobDetail; 