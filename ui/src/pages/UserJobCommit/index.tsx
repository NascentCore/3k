import { apiPostUserJob, useApiGetGpuType } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { history } from '@umijs/max';
import { Form, Input, Radio, Select, Space, Tooltip, message } from 'antd';
import React, { useState } from 'react';
import { useIntl } from '@umijs/max';
import { QuestionCircleFilled } from '@ant-design/icons';
import AsyncButton from '@/components/AsyncButton';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    stopType: '1',
    gpuNumber: 1,
    stopTime: 5,
  });
  const { data: gpuTypeOptions } = useApiGetGpuType({});

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log(values);
      return apiPostUserJob({
        data: {
          ...values,
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
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          style={{ maxWidth: 600 }}
        >
          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
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
            <Space align={'baseline'}>
              <Form.Item
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
                />
              </Form.Item>
              MB
            </Space>
          </Form.Item>

          <Form.Item
            style={{ marginBottom: 0 }}
            label={
              <>
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
            <Space align={'baseline'}>
              <Form.Item
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
                />
              </Form.Item>
              MB
            </Space>
          </Form.Item>

          <Form.Item label={'GPU'} style={{ marginBottom: 0 }}>
            <Space>
              <Form.Item
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
                <Input disabled type="number" />
              </Form.Item>
              <Form.Item
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
                  style={{ width: 200 }}
                  allowClear
                  options={
                    gpuTypeOptions?.map((x) => ({ ...x, label: x.gpuProd, value: x.gpuProd })) || []
                  }
                  placeholder={intl.formatMessage({
                    id: 'pages.UserJobCommit.form.placeholder',
                    defaultMessage: '请输入',
                  })}
                />
              </Form.Item>
            </Space>
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

          <Form.Item
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
