import { apiPostApiNode } from '@/services';
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

  const [form] = Form.useForm();
  useEffect(() => {
    form.setFieldsValue({});
  }, []);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log('Form values:', values);
      return Promise.resolve().then(() => {
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
          name="xxx"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '实例名称',
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
              id: 'xxx',
              defaultMessage: '实例名称',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="xxx"
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
              id: 'xxx',
              defaultMessage: 'CPU',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="xxx"
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
            type="text"
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'memory',
            })}
            suffix="MB"
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="xxx"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'gpu',
          })}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'gpu',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="xxx"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'gpu类型',
          })}
        >
          <Select
            allowClear
            options={[]}
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'gpu类型',
            })}
          />
        </Form.Item>

        <Form.Item
          name="xxx"
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
            type="text"
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: '数据卷大小',
            })}
            allowClear
            suffix="MB"
          />
        </Form.Item>

        <Form.Item
          name="xxx"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '挂载模型',
          })}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: '挂载模型',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="xxx"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '模型挂载路径',
          })}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: '模型挂载路径',
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
