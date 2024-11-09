/**
 * @name 推理
 * @description 推理
 */
import { apiInference, useGpuTypeOptions, useResourceAdaptersOptions } from '@/services';
import { Button, Drawer, Form, Input, Select, message, Row, Col } from 'antd';
import { useState } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

interface ContentProps {
  record: {
    id: string;
    name: string;
    category?: string;
    path?: string;
    size?: number;
    is_public?: boolean;
    template?: string;
    inference_gpu_count?: number;
  };
  onCancel: () => void;
}

interface IndexProps {
  record: {
    id: string;
    name: string;
    category?: string;
    path?: string;
    size?: number;
    is_public?: boolean;
    template?: string;
    inference_gpu_count?: number;
  };
}

const Content = ({ record, onCancel }: ContentProps) => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model_name: record?.name,
    min_instances: 1,
    max_instances: 1,
  });

  const gpuTypeOptions = useGpuTypeOptions();
  const adaptersOptions = useResourceAdaptersOptions();
  const isShowAdapter = record?.category === 'chat';

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      const minInstances = parseInt(values.min_instances, 10);
      const maxInstances = parseInt(values.max_instances, 10);

      // 检查最大实例数是否大于等于最小实例数
      if (minInstances && maxInstances && minInstances > maxInstances) {
        message.error(intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.validation.instancesError',
          defaultMessage: '最小实例数不能大于最大实例数',
        }));
        return;
      }

      // 检查实例数是否超过GPU资源限制
      const selectedGpuType = gpuTypeOptions?.find(x => x.value === values.gpu_model);
      const totalGpuCount = selectedGpuType?.gpuAllocatable || 0;
      const gpuPerInstance = record?.inference_gpu_count || 1;
      const maxPossibleInstances = Math.floor(totalGpuCount / gpuPerInstance);

      if (totalGpuCount === 0) {
        message.error(intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.validation.noGpu',
          defaultMessage: '无可用GPU',
        }));
        return;
      }

      if (maxInstances > maxPossibleInstances) {
        message.error(intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.validation.maxInstancesError',
          defaultMessage: `最大实例数不能超过 ${maxPossibleInstances}（可用GPU数量 ${totalGpuCount} / 每实例所需GPU数量 ${gpuPerInstance}）`,
        }));
        return;
      }

      setFormValues(values);
      const currentAdapter = adaptersOptions?.find((x) => x.id === values.adapter);
      const params = {
        data: {
          gpu_model: values.gpu_model,
          model_category: record?.category,
          gpu_count: record?.inference_gpu_count,
          model_id: record.id,
          model_name: record.name,
          model_path: record.path,
          model_size: record.size,
          model_is_public: record.is_public,
          model_template: record.template,
          adapter_id: currentAdapter?.id,
          adapter_name: currentAdapter?.name,
          adapter_size: currentAdapter?.size,
          adapter_is_public: currentAdapter?.is_public,
          min_instances: minInstances,
          max_instances: maxInstances,
        },
      };

      return apiInference(params).then(() => {
        message.success(
          intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.submit.success',
          }),
        );
        onCancel();
        history.push('/JobDetail');
      });
    });
  };

  return (
    <>
      <Form
        form={form}
        initialValues={formValues}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        style={{ maxWidth: 600 }}
      >
        <Form.Item
          name="model_name"
          label={intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.form.model_name',
            // defaultMessage: '模型名称',
          })}
        >
          {formValues.model_name}
        </Form.Item>

        <Form.Item
          name="adapter"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.category',
            defaultMessage: '类型',
          })}
        >
          {record?.category}
        </Form.Item>

        <Form.Item
          name="gpu_model"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
            defaultMessage: 'GPU型号',
          })}
          rules={[{ required: true }]}
        >
          <Select
            allowClear
            options={gpuTypeOptions}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
              defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        {isShowAdapter && (
          <Form.Item
            name="adapter"
            label={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.adapter',
              defaultMessage: '适配器',
            })}
          >
            <Select
              allowClear
              options={adaptersOptions}
              placeholder='无'
            />
          </Form.Item>
        )}

        <Row style={{ marginBottom: 15, marginTop: 24 }}>
          <Col span={8} style={{ textAlign: 'right' }}>
            {intl.formatMessage({
              id: 'pages.modelRepository.InferenceDrawer.form.autoScaling',
              defaultMessage: '自动扩容配置',
            })}
          </Col>
          <Col span={16}></Col>
        </Row>

        <Form.Item
          name="min_instances"
          label={intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.form.minInstances',
            defaultMessage: '最小实例数',
          })}
          rules={[
            {
              required: true,
              message: intl.formatMessage({
                id: 'pages.modelRepository.InferenceDrawer.validation.required',
                defaultMessage: '请输入最小实例数',
              }),
            }
          ]}
        >
          <Input
            type="number"
            min={1}
            step={1}
            onKeyPress={(e) => {
              if (e.key === '-' || e.key === '+' || e.key === 'e' || e.key === '.') {
                e.preventDefault();
              }
            }}
            onChange={(e) => {
              let value = e.target.value;
              if (!value || parseInt(value, 10) < 1) {
                form.setFieldValue('min_instances', 1);
              }
            }}
            onBlur={(e) => {
              let value = e.target.value;
              if (!value || value.trim() === '') {
                form.setFieldValue('min_instances', 1);
              }
            }}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.InferenceDrawer.form.minInstances.placeholder',
              defaultMessage: '请输入最小实例数',
            })}
          />
        </Form.Item>

        <Form.Item
          name="max_instances"
          label={intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.form.maxInstances',
            defaultMessage: '最大实例数',
          })}
          rules={[
            {
              required: true,
              message: intl.formatMessage({
                id: 'pages.modelRepository.InferenceDrawer.validation.required',
                defaultMessage: '请输入最大实例数',
              }),
            }
          ]}
        >
          <Input
            type="number"
            min={1}
            step={1}
            onKeyPress={(e) => {
              if (e.key === '-' || e.key === '+' || e.key === 'e' || e.key === '.') {
                e.preventDefault();
              }
            }}
            onChange={(e) => {
              let value = e.target.value;
              if (!value || parseInt(value, 10) < 1) {
                form.setFieldValue('max_instances', 1);
              }
            }}
            onBlur={(e) => {
              let value = e.target.value;
              if (!value || value.trim() === '') {
                form.setFieldValue('max_instances', 1);
              }
            }}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.InferenceDrawer.form.maxInstances.placeholder',
              defaultMessage: '请输入最大实例数',
            })}
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.modelRepository.InferenceDrawer.submit',
                // defaultMessage: '部署',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.cancel',
                // defaultMessage: '取消',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = ({ record }: IndexProps) => {
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        type="link"
        onClick={() => {
          setOpen(true);
        }}
      >
        {intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.title',
          // defaultMessage: '推理',
        })}
      </Button>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.title',
          // defaultMessage: '推理',
        })}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && <Content record={record} onCancel={() => setOpen(false)} />}
      </Drawer>
    </>
  );
};

export default Index;
