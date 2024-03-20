import { Footer } from '@/components';
import { history, useModel, Helmet } from '@umijs/max';
import { Alert } from 'antd';
import React, { useState } from 'react';
import { flushSync } from 'react-dom';
import { createStyles } from 'antd-style';
import { apiAuthLogin } from '@/services';
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
      paddingTop: 100,
    },
  };
});

const Login: React.FC = () => {
  const [type, setType] = useState<string>('login');
  const { styles } = useStyles();

  return (
    <div className={styles.container}>
      <div
        style={{
          flex: '1',
          padding: '32px 0',
        }}
      >
        {type === 'login' && <LoginForm setType={setType} />}
        {type === 'regist' && <RegistrationForm setType={setType} />}
      </div>
      <Footer />
    </div>
  );
};

export default Login;
