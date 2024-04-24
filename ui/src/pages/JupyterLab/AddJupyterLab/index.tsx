import {
  apiPostApiNode,
  apiPostJobJupyterlab,
  useApiGetGpuType,
  useApiResourceModels,
} from '@/services';
import { Button, Drawer, Form, Input, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

interface IProps {
  onChange: () => void;
  onCancel: () => void;
}

const Index = ({ onChange, onCancel }: IProps) => {
  const intl = useIntl();

  const { data: resourceModels }: any = useApiResourceModels();
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
  const { data: gpuTypeOptions } = useApiGetGpuType();
  const gpuTypeOptionsList = gpuTypeOptions?.map((x) => ({
    ...x,
    label: x.gpuProd,
    value: x.gpuProd,
  }));

  const [form] = Form.useForm();
  useEffect(() => {
    form.setFieldsValue({
      cpu_count: 2,
      memory: 2048,
      data_volume_size: 1024,
      model_path: '/model',
    });
  }, []);
  const gpu_product_watch = Form.useWatch('gpu_product', form);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log('Form values:', values);
      return apiPostJobJupyterlab({
        data: {
          ...values,
          cpu_count: Number(values.cpu_count),
          memory: Number(values.memory),
          data_volume_size: Number(values.data_volume_size),
          gpu_count: values.gpu_count ? Number(values.gpu_count) : void 0,
        },
      }).then(() => {
        onChange();
        message.success(
          intl.formatMessage({
            id: 'xxx',
            defaultMessage: '添加成功',
          }),
        );
        onCancel();
      });
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="instance_name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '实例名称',
          })}
          rules={[
            {
              required: true,
            },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message: '实例名称只能包含大小写字母、数字、下划线(_)和分割符(-)',
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="cpu_count"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'CPU',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="memory"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'memory',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            suffix="MB"
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="gpu_count"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'gpu',
          })}
          rules={
            gpu_product_watch && [
              {
                required: true,
              },
            ]
          }
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="gpu_product"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'gpu类型',
          })}
        >
          <Select
            allowClear
            options={gpuTypeOptionsList}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>

        <Form.Item
          name="data_volume_size"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '数据卷大小',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
            suffix="MB"
          />
        </Form.Item>

        <Form.Item
          name="model_id"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '挂载模型',
          })}
        >
          <Select
            allowClear
            options={resourceModelsList}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>
        <Form.Item
          name="model_path"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '模型挂载路径',
          })}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.confirm',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.cancel',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

export default Index;
