import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { LoginForm, ProFormCheckbox, ProFormText } from '@ant-design/pro-components';
import { history, useModel } from '@umijs/max';

import React from 'react';
import { flushSync } from 'react-dom';
import { apiAuthLogin } from '@/services';
import { saveToken } from '@/utils';
import { Form, Input, Button, Checkbox, Space, Typography } from 'antd';
import { Store } from 'antd/lib/form/interface';
import { apiUsersRegisterUser } from '@/services';
import { encrypt } from '@/utils/rsaEncrypt';
import AsyncButton from '@/components/AsyncButton';

const { Title, Text } = Typography;

const Login: React.FC = ({ setType }) => {
  const { initialState, setInitialState } = useModel('@@initialState');
  const [form] = Form.useForm();

  const fetchUserInfo = async () => {
    const userInfo = await initialState?.fetchUserInfo?.();

    if (userInfo) {
      flushSync(() => {
        setInitialState((s) => ({
          ...s,
          currentUser: userInfo,
        }));
      });
    }
  };

  const handleSubmit = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      // 登录
      return apiAuthLogin({
        data: {
          password: encrypt(values.password),
          username: values.username,
        },
      }).then((loginRes) => {
        saveToken(loginRes?.token);
        const urlParams = new URL(window.location.href).searchParams;
        return fetchUserInfo().then((res) => {
          history.push(urlParams.get('redirect') || '/');
        });
      });
    });
  };

  return (
    <>
      <div style={{ width: 330, margin: 'auto' }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: 30 }}>
          算想云
        </Title>
        <Form
          name="registration_form"
          form={form}
          onFinish={handleSubmit}
          initialValues={{ remember: true }}
          size={'large'}
        >
          <Form.Item name="username" rules={[{ required: true, message: '请输入邮箱' }]}>
            <Input prefix={<UserOutlined />} placeholder="请输入邮箱" allowClear />
          </Form.Item>

          <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password prefix={<LockOutlined />} placeholder="请输入密码" />
          </Form.Item>

          <Form.Item name="agree_terms" valuePropName="checked">
            <Checkbox>记住我</Checkbox>
          </Form.Item>

          <Form.Item>
            <div style={{ display: 'flex', gap: 10 }}>
              <AsyncButton type="primary" block onClick={handleSubmit}>
                登录
              </AsyncButton>
              <Button block onClick={() => setType('regist')}>
                注册
              </Button>
            </div>
          </Form.Item>

          <Form.Item style={{ paddingLeft: 80 }}>
            <div style={{ paddingTop: 10 }}>
              <Text type="secondary">客服电话：15910236560</Text>
            </div>
            <div>
              <Text type="secondary">客服邮箱：help@sxwl.ai</Text>
            </div>
          </Form.Item>
        </Form>
      </div>
    </>
  );
};

export default Login;
