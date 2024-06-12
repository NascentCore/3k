import {
  apiPostUserJob,
  useApiGetGpuType,
  useApiGetJobJupyterImage,
  useApiResourceDatasets,
  useApiResourceModels,
  useResourceDatasetsOptions,
  useResourceModelsOptions,
} from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { history } from '@umijs/max';
import { Flex, Form, Input, Radio, Select, Space, Tooltip, message } from 'antd';
import React, { useState } from 'react';
import { useIntl } from '@umijs/max';
import { QuestionCircleFilled } from '@ant-design/icons';
import AsyncButton from '@/components/AsyncButton';
import { concatArray, formatFileSize } from '@/utils';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    ckptPath: '/workspace/ckpt',
    modelPath: '/workspace/saved_model',
    stopType: 0,
    gpuNumber: 1,
    stopTime: 0,
  });
  const { data: gpuTypeOptions } = useApiGetGpuType();
  // const { data: resourceModels }: any = useApiResourceModels();
  // const { data: resourceDatasets }: any = useApiResourceDatasets();
  // const { data: jobJupyterImages }: any = useApiGetJobJupyterImage();

  const gpuTypeOptionsList = gpuTypeOptions?.map((x) => ({
    ...x,
    label: (
      <>
        <span style={{ marginRight: 20 }}>{x.gpuProd}</span>
        <span>{x.amount}元/时/个</span>
      </>
    ),
    value: x.gpuProd,
  }));
  const resourceModelsOptions = useResourceModelsOptions();
  const resourceDatasetsOption = useResourceDatasetsOptions();

  // const jobJupyterImagesList = jobJupyterImages?.data?.map((x) => ({
  //   label: x.image_name,
  //   value: x.image_name,
  // }));

  const gpuProdValue = Form.useWatch('gpuType', form);
  // console.log({ gpuTypeOptions, resourceModels, resourceDatasets, gpuProdValue });

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log(values);
      const currentModel: any = resourceModelsOptions.find((x: any) => x.value === values.model_id);
      const currentDataSet: any = resourceDatasetsOption.find(
        (x: any) => x.value === values.dataset_id,
      );
      console.log({ currentDataSet, currentModel });
      const params = {
        ...values,
        // imagePath: values.imagePath[0],
        ckptVol: Number(values.ckptVol),
        gpuNumber: Number(values.gpuNumber),
        modelVol: Number(values.modelVol),

        model_id: currentModel?.id,
        model_name: currentModel?.name,
        // model_path: currentModel?.path,
        model_size: currentModel?.size,
        model_is_public: currentModel?.is_public,
        model_template: currentModel?.template,
        //
        dataset_id: currentDataSet?.id,
        dataset_name: currentDataSet?.name,
        // dataset_path: currentDataSet?.path,
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
                  defaultMessage: 'CKPT 路径',
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
                  defaultMessage: '模型保存路径',
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
                <span style={{ color: '#ff4d4f', marginRight: 4 }}>*</span>GPU
              </>
            }
            style={{ marginBottom: 0 }}
          >
            <div style={{ display: 'flex', gap: 10 }}>
              <Form.Item
                style={{ flex: '0 0 100px' }}
                name="gpuNumber"
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
                  min={1}
                  max={
                    gpuProdValue
                      ? gpuTypeOptions?.find((x) => x.gpuProd === gpuProdValue).gpuAllocatable
                      : 1
                  }
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="gpuType"
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
                  options={gpuTypeOptionsList}
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
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
