/**
 * @name 微调
 * @description 微调
 */
import { apiFinetunes, useApiGetGpuType, useResourceDatasetsOptions } from '@/services';
import { Button, Checkbox, Col, Drawer, Form, Input, Row, Select, message } from 'antd';
import { useState } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

const Content = ({ record, onCancel }) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model: record?.name,
    // gpuProd: '',
    gpu_count: record?.finetune_gpu_count || 1,
    hyperparameters: {
      n_epochs: '3.0',
      batch_size: '4',
      learning_rate_multiplier: '5e-5',
    },
  });

  const resourceDatasetsOption = useResourceDatasetsOptions();

  const { data: gpuTypeOptions } = useApiGetGpuType({});

  const gpuProdValue = Form.useWatch('gpu_model', form);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log('Form values:', values);
      const currentModel = record;
      const currentDataSet = resourceDatasetsOption.find(
        (x: any) => x.value === values.training_file,
      );
      const params = {
        ...values,
        model_saved_type: values.model_saved_type ? 'full' : 'lora',
        gpu_count: Number(values.gpu_count),
        //
        model_id: currentModel.id,
        model_name: currentModel.name,
        model_path: currentModel.path,
        model_size: currentModel.size,
        model_is_public: currentModel.is_public,
        model_template: currentModel.template,
        //
        dataset_id: currentDataSet.id,
        dataset_name: currentDataSet.name,
        dataset_path: currentDataSet.path,
        dataset_size: currentDataSet.size,
        dataset_is_public: currentDataSet.is_public,
        //
      };
      console.log('params', params);
      return apiFinetunes({
        data: params,
      }).then((res) => {
        message.success(
          intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.submit.success',
            // defaultMessage: '微调任务创建成功',
          }),
        );
        onCancel();
        history.push('/UserJob');
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
          name="model"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.model',
            // defaultMessage: '模型名称',
          })}
        >
          {formValues.model}
        </Form.Item>
        <Form.Item
          name="training_file"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.training_file',
            // defaultMessage: '数据集',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={resourceDatasetsOption}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.training_file.placeholder',
              // defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpu_model"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
            defaultMessage: 'GPU型号',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={
              gpuTypeOptions?.map((x) => ({ ...x, label: x.gpuProd, value: x.gpuProd })) || []
            }
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
              defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpu_count"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
            defaultMessage: 'GPU数量',
          })}
          rules={[
            {
              required: true,
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
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
              defaultMessage: 'GPU数量',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="model_saved_type"
          valuePropName="checked"
          label={<div></div>}
          colon={false}
        >
          <Checkbox>微调后保存完整模型（默认保存Lora）</Checkbox>
        </Form.Item>
        <Row style={{ marginBottom: 15 }}>
          <Col span={8} style={{ textAlign: 'right' }}>
            Hyperparameters
          </Col>
          <Col span={16}></Col>
        </Row>

        <Form.Item
          name={['hyperparameters', 'n_epochs']}
          label="epochs"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
              // defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name={['hyperparameters', 'batch_size']}
          label="batch_size"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
              // defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name={['hyperparameters', 'learning_rate_multiplier']}
          label="Learning rate"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
              // defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.title',
                // defaultMessage: '微调',
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
          id: 'pages.modelRepository.fineTuningDrawer.title',
          // defaultMessage: '微调',
        })}
      </Button>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.modelRepository.fineTuningDrawer.title',
          // defaultMessage: '微调',
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
