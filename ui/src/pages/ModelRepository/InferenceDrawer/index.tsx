/**
 * @name 推理
 * @description 推理
 */
import { apiInference, useGpuTypeOptions, useResourceAdaptersOptions } from '@/services';
import { Button, Drawer, Form, Input, Select, message } from 'antd';
import { useState } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

const Content = ({ record, onCancel }) => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model_name: record.name,
    // gpuProd: '',
    // gpu_count: record?.inference_gpu_count,
  });

  const gpuTypeOptions = useGpuTypeOptions({});

  const gpuProdValue = Form.useWatch('gpu_model', form);

  const adaptersOptions = useResourceAdaptersOptions();

  // 是否显示适配器选项
  const isShowAdapter = record?.category === 'chat';

  const onFinish = () => {
    return form.validateFields().then(() => {
      const currentModel = record;
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log('Form values:', values);
      const currentAdapter = adaptersOptions?.find((x) => x.id === values.adapter);
      const params = {
        data: {
          gpu_model: values.gpu_model,
          model_category: record?.category,
          gpu_count: record?.inference_gpu_count,
          //
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
