/**
 * @name 推理
 * @description 推理
 */
import { apiGetInference, apiInference } from '@/services';
import { Button, Drawer, Form, Select, Space, Table, message } from 'antd';
import { useEffect, useState } from 'react';
import { history, Link } from '@umijs/max';

const Content = ({ record, onCancel }) => {
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({ model_name: record.id });

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
          apiInference({ data: { model_name: values.model_name } }).then((res) => {
            message.success('部署任务创建成功');
            onCancel();
            history.push('/InferenceState');
          });
        }}
      >
        <Form.Item name="model_name" label="模型名称">
          {formValues.model_name}
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <Space>
            <Button type="primary" htmlType="submit">
              部署
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = ({ record }) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        type="link"
        onClick={() => {
          setOpen(true);
        }}
      >
        推理
      </Button>
      <Drawer
        width={1000}
        title="推理"
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
