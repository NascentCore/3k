/**
 * @name 推理
 * @description 推理
 */
import { apiGetInference, apiInference, useApiGetGpuType } from '@/services';
import { Button, Drawer, Form, Input, Select, Space, Table, message } from 'antd';
import { useEffect, useState } from 'react';
import { history, Link } from '@umijs/max';
import { useIntl } from '@umijs/max';

const Content = ({ record, onCancel }) => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model_name: record.id,
    gpuProd: '',
    gpuAllocatable: 1,
  });

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
          return;
          apiInference({ data: { model_name: values.model_name } }).then((res) => {
            message.success(
              intl.formatMessage({
                id: 'pages.modelRepository.InferenceDrawer.submit.success',
                // defaultMessage: '部署任务创建成功',
              }),
            );
            onCancel();
            history.push('/InferenceState');
          });
        }}
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
              id: 'aa',
              defaultMessage: 'GPU数量',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <Space>
            <Button type="primary" htmlType="submit">
              {intl.formatMessage({
                id: 'pages.modelRepository.InferenceDrawer.submit',
                // defaultMessage: '部署',
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
