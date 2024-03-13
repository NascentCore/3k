/**
 * @name 微调
 * @description 微调
 */
import { apiFinetunes, apiResourceDatasets } from '@/services';
import { Button, Col, Drawer, Form, Input, Row, Select, Space, message } from 'antd';
import { useEffect, useState } from 'react';
import { history, Link } from '@umijs/max';

const Content = ({ record, onCancel }) => {
  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model: record?.id,
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
          apiFinetunes({ data: values }).then((res) => {
            message.success('微调任务创建成功');
            onCancel();
            history.push('/UserJob');
          });
        }}
      >
        <Form.Item name="model" label="模型名称">
          {/* <Input placeholder="请输入" allowClear readOnly /> */}
          {formValues.model}
        </Form.Item>
        <Form.Item name="training_file" label="数据集">
          <Select allowClear options={resourceDatasetsOption} placeholder="请选择" />
        </Form.Item>
        <Row style={{ marginBottom: 15 }}>
          <Col span={8} style={{ textAlign: 'right' }}>
            Hyperparameters
          </Col>
          <Col span={16}></Col>
        </Row>

        <Form.Item name={['hyperparameters', 'n_epochs']} label="epochs">
          <Input placeholder="请输入" allowClear />
        </Form.Item>
        <Form.Item name={['hyperparameters', 'batch_size']} label="batch_size">
          <Input placeholder="请输入" allowClear />
        </Form.Item>
        <Form.Item name={['hyperparameters', 'learning_rate_multiplier']} label="Learning rate">
          <Input placeholder="请输入" allowClear />
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <Space>
            <Button type="primary" htmlType="submit">
              微调
            </Button>
            <Button type="default" onClick={() => onCancel()}>
              取消
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
        微调
      </Button>
      <Drawer
        width={1000}
        title="微调"
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
