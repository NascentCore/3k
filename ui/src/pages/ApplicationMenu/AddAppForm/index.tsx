import { Button, Form, Input, Select, message } from 'antd';
import { useEffect } from 'react';
import { useIntl, useModel } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';
import { apiPostAppJob } from '@/services';

interface IProps {
  onChange: () => void;
  onCancel: () => void;
  record: any;
}

const Index = ({ onChange, onCancel, record }: IProps) => {
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const intl = useIntl();

  const [form] = Form.useForm();
  useEffect(() => {
    if (record) {
      form.setFieldsValue({});
    } else {
      form.setFieldsValue({});
    }
  }, [record]);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      const params = {
        app_id: record.app_id,
        app_name: record.app_name,
        instance_name: '',
      };
      return apiPostAppJob({ data: params }).then((res) => {
        message.success('Success');
        onChange();
      });
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          label={intl.formatMessage({
            id: 'pages.applicationMenu.appAddForm.form.app_name',
            defaultMessage: '应用名称',
          })}
        >
          {record.app_name}
        </Form.Item>
        {/* 
        <Form.Item
          name="name"
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
          name="name"
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
            suffix="GB"
          />
        </Form.Item>
        <Form.Item
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '磁盘',
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
            suffix={'GB'}
          />
        </Form.Item>
        <Form.Item
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '服务器线程数',
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
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'Embedding API',
          })}
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
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'Embedding API KEY',
          })}
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
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'LLM API',
          })}
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
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: 'LLM API KEY',
          })}
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
        */}
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.global.button.submit',
                defaultMessage: '提交',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.global.button.cancel',
                defaultMessage: '取消',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

export default Index;
