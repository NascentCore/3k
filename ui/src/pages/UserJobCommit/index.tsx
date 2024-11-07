import {
  apiPostUserJob,
  useGpuTypeOptions,
  useResourceDatasetsOptions,
  useResourceModelsOptions,
} from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { history } from '@umijs/max';
import { Form, Input, Select, Tooltip, message } from 'antd';
import React, { useState } from 'react';
import { useIntl } from '@umijs/max';
import { QuestionCircleFilled } from '@ant-design/icons';
import AsyncButton from '@/components/AsyncButton';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    ckptPath: '/workspace/ckpt',
    modelPath: '/workspace/saved_model',
    stopType: 0,
    gpuNumber: 1,
    stopTime: 0,
    nodeCount: 1,
  });
  const gpuTypeOptions = useGpuTypeOptions();
  const resourceModelsOptions = useResourceModelsOptions();
  const resourceDatasetsOption = useResourceDatasetsOptions();

  const gpuProdValue = Form.useWatch('gpuType', form);

  // 添加一个新的 watch 来监听 jobType
  const jobType = Form.useWatch('jobType', form);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      const currentModel: any = resourceModelsOptions.find((x: any) => x.value === values.model_id);
      const currentDataSet: any = resourceDatasetsOption.find(
        (x: any) => x.value === values.dataset_id,
      );
      
      const params = {
        ...values,
        nodeCount: Number(values.nodeCount),
        ckptVol: Number(values.ckptVol),
        // 只有在 General 类型且未选择 GPU Type 时，才设置默认值
        gpuNumber: (values.jobType === 'General' && !values.gpuType) ? 0 : Number(values.gpuNumber),
        gpuType: (values.jobType === 'General' && !values.gpuType) ? '' : values.gpuType,
        modelVol: Number(values.modelVol),

        model_id: currentModel?.id,
        model_name: currentModel?.name,
        model_size: currentModel?.size,
        model_is_public: currentModel?.is_public,
        model_template: currentModel?.template,

        dataset_id: currentDataSet?.id,
        dataset_name: currentDataSet?.name,
        dataset_size: currentDataSet?.size,
        dataset_is_public: currentDataSet?.is_public,
      };
      return apiPostUserJob({
        data: params,
      }).then(() => {
        message.success(
          intl.formatMessage({
            id: 'pages.UserJobCommit.form.submit.success',
            defaultMessage: '操作成功',
          }),
        );
        history.push('/UserJob');
      });
    });
  };
  return (
    <PageContainer>
      <>
        <Form
          form={form}
          initialValues={formValues}
          labelCol={{ span: 6 }}
          wrapperCol={{ span: 18 }}
          style={{ maxWidth: 800 }}
        >
          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
                <span style={{ color: '#ff4d4f', marginRight: 4 }}>*</span>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.ckptPath',
                  defaultMessage: '工作目录',
                })}
                <Tooltip
                  title={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.ckptPath.tooltip',
                    defaultMessage: '',
                  })}
                >
                  <QuestionCircleFilled style={{ marginLeft: 6 }} />
                </Tooltip>
              </>
            }
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <Form.Item
                style={{ flex: 1 }}
                name="ckptPath"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                  {
                    pattern: /^\//,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.path.error',
                      defaultMessage: '路径必须以"/"开头',
                    }),
                    whitespace: true,
                  },
                ]}
              >
                <Input
                  allowClear
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="ckptVol"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.ckptVol',
                      defaultMessage: '容量',
                    })}
                    <Tooltip
                      title={intl.formatMessage({
                        id: 'pages.UserJobCommit.form.ckptVol.tooltip',
                        defaultMessage: '',
                      })}
                    >
                      <QuestionCircleFilled style={{ marginLeft: 6 }} />
                    </Tooltip>
                  </>
                }
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                  suffix="MB"
                />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
                <span style={{ color: '#ff4d4f', marginRight: 4 }}>*</span>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.modelPath',
                  defaultMessage: '输出目录',
                })}
                <Tooltip
                  title={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.modelPath.tooltip',
                    defaultMessage: '',
                  })}
                >
                  <QuestionCircleFilled style={{ marginLeft: 6 }} />
                </Tooltip>
              </>
            }
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <Form.Item
                style={{ flex: 1 }}
                name="modelPath"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                  {
                    pattern: /^\//,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.path.error',
                      defaultMessage: '路径必须以"/"开头',
                    }),
                    whitespace: true,
                  },
                ]}
              >
                <Input
                  allowClear
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="modelVol"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.modelVol',
                      defaultMessage: '容量',
                    })}
                    <Tooltip
                      title={intl.formatMessage({
                        id: 'pages.UserJobCommit.form.modelVol.tooltip',
                        defaultMessage: '',
                      })}
                    >
                      <QuestionCircleFilled style={{ marginLeft: 6 }} />
                    </Tooltip>
                  </>
                }
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                  suffix="MB"
                />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item
            label={
              <>
                <span style={{ color: '#ff4d4f', marginRight: 4 }}>*</span>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.nodeCount',
                  defaultMessage: '节点数量',
                })}
              </>
            }
            style={{ marginBottom: 0 }}
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <Form.Item
                style={{ flex: '0 0 100px' }}
                name="nodeCount"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                ]}
              >
                <Input type="number" min={1} />
              </Form.Item>

              <Form.Item
                style={{ flex: 1 }}
                label="GPU"
                required
              >
                <div style={{ display: 'flex', gap: 10 }}>
                  <Form.Item
                    style={{ flex: '0 0 100px', marginBottom: 0 }}
                    name="gpuNumber"
                    rules={[
                      {
                        required: jobType !== 'General',
                        message: intl.formatMessage({
                          id: 'pages.UserJobCommit.form.placeholder',
                          defaultMessage: '请输入',
                        }),
                      },
                    ]}
                  >
                    <Input
                      type="number"
                      min={1}
                      max={
                        gpuProdValue
                          ? Math.max(
                              ...gpuTypeOptions
                                .filter((x) => x.gpuProd === gpuProdValue)
                                .map((x) => x.gpuAllocatable)
                            )
                          : 1
                      }
                    />
                  </Form.Item>
                  <Form.Item
                    style={{ flex: 1, marginBottom: 0 }}
                    name="gpuType"
                    rules={[
                      {
                        required: jobType !== 'General',
                        message: intl.formatMessage({
                          id: 'pages.UserJobCommit.form.placeholder',
                          defaultMessage: '请输入',
                        }),
                      },
                    ]}
                  >
                    <Select
                      allowClear
                      options={gpuTypeOptions}
                      placeholder={intl.formatMessage({
                        id: 'pages.UserJobCommit.form.placeholder',
                        defaultMessage: '请输入',
                      })}
                    />
                  </Form.Item>
                </div>
              </Form.Item>
            </div>
          </Form.Item>
          <Form.Item
            name="imagePath"
            label={
              <>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.imagePath',
                  defaultMessage: '容器镜像',
                })}
                <Tooltip
                  title={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.imagePath.tooltip',
                    defaultMessage: '',
                  })}
                >
                  <QuestionCircleFilled style={{ marginLeft: 6 }} />
                </Tooltip>
              </>
            }
            rules={[
              {
                required: true,
                message: intl.formatMessage({
                  id: 'pages.UserJobCommit.form.placeholder',
                  defaultMessage: '请输入',
                }),
              },
            ]}
          >
            {/* <Select
              mode="tags"
              maxCount={1}
              options={jobJupyterImagesList}
              allowClear
              placeholder={intl.formatMessage({
                id: 'pages.UserJobCommit.form.placeholder',
                defaultMessage: '请输入',
              })}
            ></Select> */}
            <Input
              placeholder={intl.formatMessage({
                id: 'pages.UserJobCommit.form.placeholder',
                defaultMessage: '请输入',
              })}
            />
          </Form.Item>

          <Form.Item
            name="jobType"
            label={intl.formatMessage({
              id: 'pages.UserJobCommit.form.jobType',
              defaultMessage: '任务类型',
            })}
            rules={[
              {
                required: true,
                message: intl.formatMessage({
                  id: 'pages.UserJobCommit.form.placeholder',
                  defaultMessage: '请输入',
                }),
              },
            ]}
          >
            <Select
              allowClear
              options={[
                { label: 'MPI', value: 'MPI' },
                { label: 'Pytorch', value: 'Pytorch' },
                { label: 'TensorFlow', value: 'TensorFlow' },
                { label: 'General', value: 'General' },
              ]}
              placeholder={intl.formatMessage({
                id: 'pages.UserJobCommit.form.placeholder',
                defaultMessage: '请输入',
              })}
            />
          </Form.Item>

          {/* <Form.Item
            label={
              <>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.stopType',
                  defaultMessage: '终止条件',
                })}
                <Tooltip
                  title={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.stopType.tooltip',
                    defaultMessage: '',
                  })}
                >
                  <QuestionCircleFilled style={{ marginLeft: 6 }} />
                </Tooltip>
              </>
            }
          >
            <Space align={'baseline'}>
              <Form.Item name="stopType" style={{ width: 230 }}>
                <Radio.Group disabled>
                  <Radio value="0">
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.Voluntary',
                      defaultMessage: '自然终止',
                    })}
                  </Radio>
                  <Radio value="1">
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.SetDuration',
                      defaultMessage: '设定时长',
                    })}
                  </Radio>
                </Radio.Group>
              </Form.Item>
              <Form.Item
                name="stopTime"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.placeholder',
                      defaultMessage: '请输入',
                    }),
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <div style={{ width: 50 }}>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.stopTime.unit',
                  defaultMessage: '分钟',
                })}
              </div>
            </Space>
          </Form.Item> */}

          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.pretrainedModelId',
                  defaultMessage: '基座模型',
                })}
              </>
            }
          >
            <div style={{ display: 'flex', gap: 10 }}>
              <Form.Item style={{ flex: 1 }} name="model_id">
                <Select
                  allowClear
                  options={resourceModelsOptions}
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="model_path"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.pretrainedModelPath',
                      defaultMessage: '挂载路径',
                    })}
                  </>
                }
                rules={[
                  {
                    pattern: /^\//,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.path.error',
                      defaultMessage: '路径必须以"/"开头',
                    }),
                    whitespace: true,
                  },
                ]}
              >
                <Input
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.datasetId',
                  defaultMessage: '数据集',
                })}
              </>
            }
          >
            <div style={{ display: 'flex', gap: 10 }}>
              <Form.Item style={{ flex: 1 }} name="dataset_id">
                <Select
                  allowClear
                  options={resourceDatasetsOption}
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="dataset_path"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.datasetPath',
                      defaultMessage: '挂载路径',
                    })}
                  </>
                }
                rules={[
                  {
                    pattern: /^\//,
                    message: intl.formatMessage({
                      id: 'pages.UserJobCommit.form.path.error',
                      defaultMessage: '路径必须以"/"开头',
                    }),
                    whitespace: true,
                  },
                ]}
              >
                <Input
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
            </div>
          </Form.Item>

          <Form.Item
            name="runCommand"
            label={
              <>
                {intl.formatMessage({
                  id: 'pages.UserJobCommit.form.runCommand',
                  defaultMessage: '启动命令',
                })}
              </>
            }
          >
            <Input
              placeholder={intl.formatMessage({
                id: 'pages.UserJobCommit.form.placeholder',
                defaultMessage: '请输入',
              })}
            />
          </Form.Item>

          <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
            <AsyncButton type="primary" onClick={onFinish} block>
              {intl.formatMessage({
                id: 'pages.UserJobCommit.form.submit',
                defaultMessage: '提交',
              })}
            </AsyncButton>
          </Form.Item>
        </Form>
      </>
    </PageContainer>
  );
};

export default Welcome;
