import { Footer } from '@/components';
import { history, useModel, Helmet, SelectLang, useLocation } from '@umijs/max';
import { Alert } from 'antd';
import React, { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { createStyles } from 'antd-style';
import { apiAuthLogin, apiGetDingtalkUserInfo } from '@/services';
import { encrypt } from '@/utils/rsaEncrypt';
import { saveToken } from '@/utils';
import LoginForm from './LoginForm';
import RegistrationForm from './RegistrationForm';

const useStyles = createStyles(({ token }) => {
  return {
    container: {
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflow: 'auto',
      backgroundImage:
        "url('https://mdn.alipayobjects.com/yuyan_qk0oxh/afts/img/V-_oS6r-i7wAAAAAAAAAAAAAFl94AQBr')",
      backgroundSize: '100% 100%',
      // paddingTop: 100,
      justifyContent: 'center',
    },
    lang: {
      width: 42,
      height: 42,
      lineHeight: '42px',
      position: 'fixed',
      right: 16,
      borderRadius: token.borderRadius,
      ':hover': {
        backgroundColor: token.colorBgTextHover,
      },
    },
  };
});
const Lang = () => {
  const { styles } = useStyles();

  return (
    <div className={styles.lang} data-lang>
      {SelectLang && <SelectLang />}
    </div>
  );
};

const Login: React.FC = () => {
  const { initialState, setInitialState } = useModel('@@initialState');
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

  const [type, setType] = useState<string>('login');
  const { styles } = useStyles();

  const { search } = useLocation();
  useEffect(() => {
    const code = new URLSearchParams(search).get('code');
    if (code) {
      console.log('dingding code', code);
      // 调用接口获取登录信息
      apiGetDingtalkUserInfo(code).then((res) => {
        console.log('dingding res', res);
        // 获取 token
        const token = '';
        saveToken(token);
        // 进入首页
        return fetchUserInfo().then((res) => {
          history.push('/');
        });
      });
    }
  }, []);

  return (
    <>
      <Lang />
      <div className={styles.container}>
        <div>
          {type === 'login' && <LoginForm setType={setType} />}
          {type === 'regist' && <RegistrationForm setType={setType} />}
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Login;
