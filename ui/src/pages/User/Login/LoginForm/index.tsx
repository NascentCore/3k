import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { LoginForm, ProFormCheckbox, ProFormText } from '@ant-design/pro-components';
import { history, useModel, useIntl } from '@umijs/max';

import React, { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { apiAuthLogin, apiCodeSendEmail } from '@/services';
import { saveToken } from '@/utils';
import { Form, Input, Button, Checkbox, Space, Typography, message } from 'antd';
import { Store } from 'antd/lib/form/interface';
import { apiUsersRegisterUser } from '@/services';
import { encrypt } from '@/utils/rsaEncrypt';
import AsyncButton from '@/components/AsyncButton';

const { Title, Text } = Typography;

enum ILoginType {
  CODE = 'CODE',
  PASSWORD = 'PASSWORD',
}

const Login: React.FC = ({ setType }) => {
  const intl = useIntl();
  const { initialState, setInitialState } = useModel('@@initialState');
  const [form] = Form.useForm();
  const [loginType, setLoginType] = useState<ILoginType>();

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
          username: values.username,
          password: values.password ? encrypt(values.password) : void 0,
          code: values.code ? values.code : void 0,
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

  const [countdown, setCountdown] = useState(0);
  // useEffect 用于处理倒计时逻辑
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [countdown]);

  const handleSendVerificationCode = async () => {
    // 模拟发送验证码的操作
    const email = form.getFieldValue('username');
    if (email && email.trim() !== '') {
      console.log('发送验证码', email);
      setCountdown(60); // 设置倒计时初始值为60秒
      apiCodeSendEmail(email);
    } else {
      message.error(
        intl.formatMessage({
          id: 'pages.regist.username',
          // defaultMessage: '请输入邮箱',
        }),
      );
    }
  };

  return (
    <>
      <div style={{ width: 330, margin: 'auto' }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: 30 }}>
          {intl.formatMessage({
            id: 'pages.login.title',
            // defaultMessage: '算想云',
          })}
        </Title>
        <Form
          name="registration_form"
          form={form}
          onFinish={handleSubmit}
          initialValues={{ remember: true }}
          size={'large'}
        >
          <Form.Item
            name="username"
            rules={[
              {
                required: true,
                message: intl.formatMessage({
                  id: 'pages.login.username',
                  // defaultMessage: '请输入邮箱',
                }),
              },
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder={intl.formatMessage({
                id: 'pages.login.username',
                // defaultMessage: '请输入邮箱',
              })}
              allowClear
            />
          </Form.Item>
          {loginType === ILoginType.PASSWORD ? (
            <>
              <Form.Item
                name="password"
                rules={[
                  {
                    required: true,
                    message: intl.formatMessage({
                      id: 'pages.login.password',
                      defaultMessage: '请输入密码',
                    }),
                  },
                ]}
              >
                <Input.Password
                  prefix={<LockOutlined />}
                  placeholder={intl.formatMessage({
                    id: 'pages.login.password',
                    defaultMessage: '请输入密码',
                  })}
                />
              </Form.Item>
            </>
          ) : (
            <>
              <Space align={'baseline'}>
                <Form.Item
                  name="code"
                  rules={[
                    {
                      required: true,
                      message: intl.formatMessage({
                        id: 'pages.regist.codemes',
                        defaultMessage: '请输入邮箱验证码',
                      }),
                    },
                  ]}
                >
                  <Input
                    placeholder={intl.formatMessage({
                      id: 'pages.regist.codemes',
                      defaultMessage: '请输入邮箱验证码',
                    })}
                    allowClear
                  />
                </Form.Item>

                <Button onClick={handleSendVerificationCode} disabled={countdown > 0}>
                  {countdown > 0
                    ? `${countdown} ${intl.formatMessage({
                        id: 'pages.regist.codemes.tip',
                        // defaultMessage: '秒后重新发送',
                      })}`
                    : intl.formatMessage({
                        id: 'pages.regist.codemes.tip2',
                        // defaultMessage: '获取验证码',
                      })}
                </Button>
              </Space>
            </>
          )}
          <Form.Item name="rememberMe" valuePropName="checked">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Checkbox>
                {intl.formatMessage({
                  id: 'pages.login.rememberMe',
                  // defaultMessage: '记住我',
                })}
              </Checkbox>
              {loginType === ILoginType.PASSWORD ? (
                <Typography.Link onClick={() => setLoginType(ILoginType.CODE)}>
                  {intl.formatMessage({
                    id: 'pages.login.loginType.code',
                    defaultMessage: '使用验证码登录',
                  })}
                </Typography.Link>
              ) : (
                <Typography.Link onClick={() => setLoginType(ILoginType.PASSWORD)}>
                  {intl.formatMessage({
                    id: 'pages.login.loginType.password',
                    defaultMessage: '使用密码登录',
                  })}
                </Typography.Link>
              )}
              <Typography.Link
                onClick={() => {
                  // 这里填写钉钉扫码登录的目标链接
                  window.location.href = 'https://oapi.dingtalk.com/connect/qrconnect?appid=dingj9mplqbu3wpn8rzn&response_type=code&scope=snsapi_login&state=STATE&redirect_uri=https://llm.nascentcore.net/user/login';
                }}
              >
                {intl.formatMessage({
                  id: 'pages.login.loginType.dingding',
                  defaultMessage: '钉钉扫码登录',
                })}
              </Typography.Link>
            </div>
          </Form.Item>

          <Form.Item>
            <div style={{ display: 'flex', gap: 10 }}>
              <AsyncButton type="primary" block onClick={handleSubmit}>
                {intl.formatMessage({
                  id: 'pages.login.submit',
                  defaultMessage: '登录',
                })}
              </AsyncButton>
              <Button block onClick={() => setType('regist')}>
                {intl.formatMessage({
                  id: 'pages.login.regist',
                  defaultMessage: '注册',
                })}
              </Button>
            </div>
          </Form.Item>

          <Form.Item style={{ paddingLeft: 80 }}>
            <div style={{ paddingTop: 10 }}>
              <Text type="secondary">
                {intl.formatMessage({
                  id: 'pages.login.servicePhone',
                  // defaultMessage: '客服电话',
                })}
              </Text>
            </div>
            <div>
              <Text type="secondary">
                {intl.formatMessage({
                  id: 'pages.login.serviceEmail',
                  // defaultMessage: '客服邮箱',
                })}
              </Text>
            </div>
          </Form.Item>
        </Form>
      </div>
    </>
  );
};

export default Login;
