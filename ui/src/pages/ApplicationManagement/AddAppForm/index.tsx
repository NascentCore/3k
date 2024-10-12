import { Button, Form, Input, Select, message } from 'antd';
import { useEffect } from 'react';
import { useIntl, useModel } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';
import { apiPostAppRegister } from '@/services';

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
      // 编辑
      form.setFieldsValue({ ...record });
    } else {
      // 新增
      form.setFieldsValue({});
    }
  }, [record]);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      if (record) {
        // 编辑
        const params = { ...values, id: record.id };
        // apiPostAppRegister({ data: params });
      } else {
        // 新增
        const params = { ...values };
        apiPostAppRegister({ data: params }).then((res) => {
          onChange();
        });
      }
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="name"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '应用名称',
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
          name="desc"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '应用描述',
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
          name="crd"
          label={intl.formatMessage({
            id: 'xxx',
            defaultMessage: '应用配置',
          })}
          rules={[{ required: true }]}
        >
          <Input.TextArea
            rows={15}
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
                defaultMessage: '提交',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'xxx',
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
