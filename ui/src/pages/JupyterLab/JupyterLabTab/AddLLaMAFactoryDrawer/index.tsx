import { Button, Drawer, Form, Input, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';
import {
  useGpuTypeOptions,
  useResourceAdaptersOptions,
  useResourceDatasetsOptions,
  useResourceModelsOptions,
} from '@/services';

interface IProps {
  onCancel: () => void;
}

const Content = ({ onCancel }: IProps) => {
  const intl = useIntl();

  const [form] = Form.useForm();

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log('Form values:', values);
      return;
    });
  };

  const gpuTypeOptions = useGpuTypeOptions();
  const modelsOptions = useResourceModelsOptions();
  const datasetsOptions = useResourceDatasetsOptions();
  const adaptersOptions = useResourceAdaptersOptions();

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '实例名称',
          })}
          rules={[{ required: true }]}
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
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'CPU',
          })}
          rules={[{ required: true }]}
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
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'MEM',
          })}
          rules={[{ required: true }]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
            min={1}
            suffix="MB"
          />
        </Form.Item>

        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'GPU数量',
          })}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
            min={1}
          />
        </Form.Item>

        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'GPU类型',
          })}
        >
          <Select
            allowClear
            options={gpuTypeOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>

        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '数据卷大小',
          })}
          rules={[{ required: true }]}
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
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '挂载模型',
          })}
          rules={[{ required: true }]}
        >
          <Select
            allowClear
            options={modelsOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>
        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '挂载适配器',
          })}
          rules={[{ required: true }]}
        >
          <Select
            allowClear
            options={adaptersOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>
        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '挂载数据集',
          })}
          rules={[{ required: true }]}
        >
          <Select
            allowClear
            options={datasetsOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'xxx',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'xxx',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = () => {
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button type={'link'} onClick={() => setOpen(true)}>
        {intl.formatMessage({
          id: 'xxx',
          defaultMessage: 'LLaMA-Factory',
        })}
      </Button>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'xxx',
          defaultMessage: '添加实例',
        })}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && <Content onCancel={() => setOpen(false)} />}
      </Drawer>
    </>
  );
};

export default Index;
