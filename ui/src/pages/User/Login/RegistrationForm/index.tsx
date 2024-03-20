import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Checkbox, Space, Typography, message } from 'antd';
import { apiCodeSendEmail, apiUsersRegisterUser } from '@/services';
import { encrypt } from '@/utils/rsaEncrypt';
import { LockOutlined, UserOutlined } from '@ant-design/icons';
import AsyncButton from '@/components/AsyncButton';
const { Title } = Typography;

const RegistrationForm: React.FC = ({ setType }) => {
  const [form] = Form.useForm();
  const [countdown, setCountdown] = useState(0);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const formValues = form.getFieldsValue();
      console.log('Received values of form: ', formValues);
      const prarms = {
        username: formValues.email,
        email: formValues.email,
        enabled: 1,
        password: encrypt(formValues.password),
      };
      console.log('prarms: ', prarms);
      return apiUsersRegisterUser(formValues.codemes, {
        data: prarms,
      }).then((res) => {
        message.success('注册成功');
        setType('login');
      });
    });
  };

  const handleSendVerificationCode = async () => {
    // 模拟发送验证码的操作
    const email = form.getFieldValue('email');
    if (email && email.trim() !== '') {
      console.log('发送验证码', email);
      setCountdown(60); // 设置倒计时初始值为60秒
      apiCodeSendEmail(email);
    }
  };

  // useEffect 用于处理倒计时逻辑
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [countdown]);

  return (
    <div style={{ width: 330, margin: 'auto' }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: 30 }}>
        用户注册
      </Title>
      <Form
        name="registration_form"
        form={form}
        onFinish={onFinish}
        initialValues={{ remember: true }}
        size={'large'}
      >
        <Form.Item name="email" rules={[{ required: true, message: '请输入邮箱' }]}>
          <Input prefix={<UserOutlined />} placeholder="请输入邮箱" allowClear />
        </Form.Item>

        <Space align={'baseline'}>
          <Form.Item name="codemes" rules={[{ required: true, message: '请输入邮箱验证码' }]}>
            <Input placeholder="请输入邮箱验证码" allowClear />
          </Form.Item>

          <Button onClick={handleSendVerificationCode} disabled={countdown > 0}>
            {countdown > 0 ? `${countdown} 秒后重新发送` : '获取验证码'}
          </Button>
        </Space>
        <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="请输入密码" />
        </Form.Item>

        <Form.Item
          name="confirm_password"
          dependencies={['password']}
          hasFeedback
          rules={[
            { required: true, message: '请确认密码' },
            ({ getFieldValue }) => ({
              validator(rule, value) {
                if (!value || getFieldValue('password') === value) {
                  return Promise.resolve();
                }
                return Promise.reject('两次输入的密码不一致');
              },
            }),
          ]}
        >
          <Input.Password prefix={<LockOutlined />} placeholder="确认密码" />
        </Form.Item>

        <Form.Item
          name="agree_terms"
          valuePropName="checked"
          rules={[
            {
              validator: (_, value) =>
                value ? Promise.resolve() : Promise.reject('请同意用户协议'),
            },
          ]}
        >
          <Checkbox>我同意算想未来隐私政策</Checkbox>
        </Form.Item>

        <Form.Item>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              注册
            </AsyncButton>
            <Button block onClick={() => setType('login')}>
              返回
            </Button>
          </div>
        </Form.Item>
      </Form>
    </div>
  );
};

export default RegistrationForm;
