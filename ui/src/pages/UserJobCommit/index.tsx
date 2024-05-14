import {
  apiPostUserJob,
  useApiGetGpuType,
  useApiResourceDatasets,
  useApiResourceModels,
} from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { history } from '@umijs/max';
import { Flex, Form, Input, Radio, Select, Space, Tooltip, message } from 'antd';
import React, { useState } from 'react';
import { useIntl } from '@umijs/max';
import { QuestionCircleFilled } from '@ant-design/icons';
import AsyncButton from '@/components/AsyncButton';
import { formatFileSize } from '@/utils';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    stopType: 0,
    gpuNumber: 1,
    stopTime: 0,
  });
  const { data: gpuTypeOptions } = useApiGetGpuType();
  const { data: resourceModels }: any = useApiResourceModels();
  const { data: resourceDatasets }: any = useApiResourceDatasets();

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
  const resourceModelsList = resourceModels?.map((x) => ({
    ...x,
    label: (
      <>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12,
          }}
        >
          <span style={{ marginRight: 20 }}>{x.name}</span>
          {/* <span>{formatFileSize(x.size)}</span> */}
        </div>
      </>
    ),
    value: x.id,
    key: x.id,
  }));
  const resourceDatasetsList = resourceDatasets?.map((x) => ({
    ...x,
    label: (
      <>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12,
          }}
        >
          <span style={{ marginRight: 20 }}>{x.name}</span>
          {/* <span>{formatFileSize(x.size)}</span> */}
        </div>
      </>
    ),
    value: x.id,
  }));

  const gpuProdValue = Form.useWatch('gpuType', form);
  // console.log({ gpuTypeOptions, resourceModels, resourceDatasets, gpuProdValue });

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log(values);
      return apiPostUserJob({
        data: {
          ...values,
          ckptVol: Number(values.ckptVol),
          gpuNumber: Number(values.gpuNumber),
          modelVol: Number(values.modelVol),
          // pretrainedModelId datasetId
          modelIsPublic: resourceModelsList.find((x: any) => x.value === values.pretrainedModelId)
            ?.is_public,
          datasetIsPublic: resourceDatasetsList.find((x: any) => x.value === values.datasetId)
            ?.is_public,
        },
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
            <Input
              allowClear
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
              <Form.Item style={{ flex: 1 }} name="pretrainedModelId">
                <Select
                  allowClear
                  options={resourceModelsList}
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="pretrainedModelPath"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.pretrainedModelPath',
                      defaultMessage: '挂载路径',
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
              <Form.Item style={{ flex: 1 }} name="datasetId">
                <Select
                  allowClear
                  options={resourceDatasetsList}
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
              <Form.Item
                style={{ flex: 1 }}
                name="datasetPath"
                label={
                  <>
                    {intl.formatMessage({
                      id: 'pages.UserJobCommit.form.datasetPath',
                      defaultMessage: '挂载路径',
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
