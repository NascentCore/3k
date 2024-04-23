import { Button, Form, Select, message } from 'antd';
import { useEffect } from 'react';
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
            defaultMessage: '操作成功',
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
            defaultMessage: '基座镜像',
          })}
        >
          <Select
            allowClear
            options={[]}
            placeholder={intl.formatMessage({
              id: 'xxx',
              defaultMessage: '基座镜像',
            })}
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <p>构建镜像时会执行如下逻辑：</p>
          <p>1. 将 /workspace 目录下的内容完整复制到镜像相同路径下</p>
          <p>2. 自动安装 /workspace 目录下的 requirements.txt</p>
          <p>请将代码及 requirements.txt 文件放到该路径下</p>
          <p>注：数据卷默认挂载路径为 /workspace</p>
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
