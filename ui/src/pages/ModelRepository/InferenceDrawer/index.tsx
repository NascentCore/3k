/**
 * @name 推理
 * @description 推理
 */
import { apiInference, useGpuTypeOptions, useResourceAdaptersOptions } from '@/services';
import { Button, Drawer, Form, Input, Select, message, Row, Col } from 'antd';
import { useState, useEffect } from 'react';
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

const Content = ({ record, onCancel }: ContentProps) => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [maxPossibleInstances, setMaxPossibleInstances] = useState(10);
  const [formValues, setFormValues] = useState({
    model_name: record?.name,
    // gpuProd: '',
    // gpu_count: record?.inference_gpu_count,
    min_instances: 1,
    max_instances: 1,
  });

  // 监听 GPU 型号的变化
  const selectedGpuModel = Form.useWatch('gpu_model', form);
  const minInstances = Form.useWatch('min_instances', form);
  const maxInstances = Form.useWatch('max_instances', form);

  // 当 GPU 型号变化时，重新计算最大可能实例数并调整表单值
  useEffect(() => {
    if (selectedGpuModel) {
      const selectedGpuType = gpuTypeOptions?.find(x => x.value === selectedGpuModel);
      const totalGpuCount = selectedGpuType?.gpuAllocatable || 0;
      const gpuPerInstance = record?.inference_gpu_count || 1;
      const newMaxPossibleInstances = Math.floor(totalGpuCount / gpuPerInstance);
      
      setMaxPossibleInstances(newMaxPossibleInstances);

      // 调整当前的实例数值
      const currentMin = form.getFieldValue('min_instances');
      const currentMax = form.getFieldValue('max_instances');
      
      if (currentMax && currentMax > newMaxPossibleInstances) {
        form.setFieldValue('max_instances', newMaxPossibleInstances);
      }
      
      if (currentMin && currentMin > newMaxPossibleInstances) {
        form.setFieldValue('min_instances', newMaxPossibleInstances);
      }
    }
  }, [selectedGpuModel, form, gpuTypeOptions, record?.inference_gpu_count]);

  // 当最大实例数变化时，确保最小实例数不大于最大实例数
  useEffect(() => {
    if (maxInstances && minInstances && minInstances > maxInstances) {
      form.setFieldValue('min_instances', maxInstances);
    }
  }, [maxInstances, minInstances, form]);

  // 当最小实例数变化时，确保最大实例数不小于最小实例数
  useEffect(() => {
    if (minInstances && maxInstances && maxInstances < minInstances) {
      form.setFieldValue('max_instances', minInstances);
    }
  }, [minInstances, maxInstances, form]);

  const gpuTypeOptions = useGpuTypeOptions({});

  const gpuProdValue = Form.useWatch('gpu_model', form);

  const adaptersOptions = useResourceAdaptersOptions();

  // 是否显示适配器选项
  const isShowAdapter = record?.category === 'chat';

  const onFinish = () => {
    return form.validateFields().then(() => {
      const currentModel = record;
      const values = form.getFieldsValue();
      
      // 检查实例数的合法性
      const minInstances = parseInt(values.min_instances, 10);
      const maxInstances = parseInt(values.max_instances, 10);
      
      if (minInstances && maxInstances && minInstances > maxInstances) {
        message.error(intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.validation.instancesError',
          defaultMessage: '最小实例数不能大于最大实例数',
        }));
        return;
      }

      // 计算单个实例需要的 GPU 数量
      const gpuPerInstance = record?.inference_gpu_count || 1;
      
      // 获取选中的 GPU 型号的总数量
      const selectedGpuType = gpuTypeOptions?.find(x => x.value === values.gpu_model);
      const totalGpuCount = selectedGpuType?.gpuAllocatable || 0;
      
      // 计算最大可能的实例数
      const maxPossibleInstances = Math.floor(totalGpuCount / gpuPerInstance);
      
      if (maxInstances && maxInstances > maxPossibleInstances) {
        message.error(intl.formatMessage({
          id: 'pages.modelRepository.InferenceDrawer.validation.maxInstancesError',
          defaultMessage: `最大实例数不能超过 ${maxPossibleInstances}（可用GPU数量 ${totalGpuCount} / 每实例所需GPU数量 ${gpuPerInstance}）`,
        }));
        return;
      }

      setFormValues(values);
      console.log('Form values:', values);
      const currentAdapter = adaptersOptions?.find((x) => x.id === values.adapter);
      const params = {
        data: {
          gpu_model: values.gpu_model,
          model_category: record?.category,
          gpu_count: record?.inference_gpu_count,
          model_id: currentModel.id,
          model_name: currentModel.name,
          model_path: currentModel.path,
          model_size: currentModel.size,
          model_is_public: currentModel.is_public,
          model_template: currentModel.template,
          adapter_id: currentAdapter?.id,
          adapter_name: currentAdapter?.name,
          adapter_size: currentAdapter?.size,
          adapter_is_public: currentAdapter?.is_public,
          min_instances: values.min_instances ? parseInt(values.min_instances, 10) : undefined,
          max_instances: values.max_instances ? parseInt(values.max_instances, 10) : undefined,
        },
      };
      console.log('Form params:', params);
      return apiInference(params).then((res) => {
        message.success(
          intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.submit.success',
            // defaultMessage: '部署任务创建成功',
          }),
        );
        onCancel();
        history.push('/InferenceState');
      });
    });

    // return;
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

        {/* <Form.Item
          name="gpu_count"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
            defaultMessage: 'GPU数量',
          })}
          rules={[{ required: true }]}
        >
          <Input
            type="number"
            min={1}
            max={
              gpuProdValue
                ? gpuTypeOptions?.find((x) => x.gpuProd === gpuProdValue).gpuAllocatable
                : 1
            }
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
              defaultMessage: 'GPU数量',
            })}
            allowClear
          />
        </Form.Item> */}

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
        >
          <Input
            type="number"
            min={1}
            max={maxPossibleInstances}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.InferenceDrawer.form.minInstances.placeholder',
              defaultMessage: '请输入最小实例数',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item
          name="max_instances"
          label={intl.formatMessage({
            id: 'pages.modelRepository.InferenceDrawer.form.maxInstances',
            defaultMessage: '最大实例数',
          })}
        >
          <Input
            type="number"
            min={1}
            max={maxPossibleInstances}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.InferenceDrawer.form.maxInstances.placeholder',
              defaultMessage: '请输入最大实例数',
            })}
            allowClear
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

const Index = ({ record }) => {
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
