/**
 * @name 微调
 * @description 微调
 */
import { apiFinetunes, apiResourceDatasets, useApiGetGpuType } from '@/services';
import { Button, Col, Drawer, Form, Input, Row, Select, Space, message } from 'antd';
import { useEffect, useState } from 'react';
import { history, Link } from '@umijs/max';
import { useIntl } from '@umijs/max';

const Content = ({ record, onCancel }) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model: record?.id,
    gpuProd: '',
    gpuAllocatable: 1,
    hyperparameters: {
      n_epochs: '3.0',
      batch_size: '4',
      learning_rate_multiplier: '5e-5',
    },
  });

  const [resourceDatasetsOption, setResourceDatasets] = useState([]);
  useEffect(() => {
    apiResourceDatasets({}).then((res) => {
      setResourceDatasets(res?.map((x) => ({ ...x, label: x.id, value: x.id })));
    });
  }, []);

  const { data: gpuTypeOptions } = useApiGetGpuType({});

  const gpuProdValue = Form.useWatch('gpuProd', form);

  return (
    <>
      <Form
        form={form}
        initialValues={formValues}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        style={{ maxWidth: 600 }}
        onFinish={(values) => {
          console.log('Form values:', values);
          setFormValues(values);
          // return;
          apiFinetunes({ data: values }).then((res) => {
            message.success(
              intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.submit.success',
                // defaultMessage: '微调任务创建成功',
              }),
            );
            onCancel();
            history.push('/UserJob');
          });
        }}
      >
        <Form.Item
          name="model"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.model',
            // defaultMessage: '模型名称',
          })}
        >
          {/* <Input placeholder="请输入" allowClear readOnly /> */}
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
          name="gpuProd"
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
            options={gpuTypeOptions?.map((x) => ({ ...x, label: x.gpuProd, value: x.gpuProd }))}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
              defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpuAllocatable"
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
          <Space>
            <Button type="primary" htmlType="submit">
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.title',
                // defaultMessage: '微调',
              })}
            </Button>
            <Button type="default" onClick={() => onCancel()}>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.cancel',
                // defaultMessage: '取消',
              })}
            </Button>
          </Space>
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
