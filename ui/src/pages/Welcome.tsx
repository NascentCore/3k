import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { theme } from 'antd';
import React from 'react';

const Welcome: React.FC = () => {
  const { token } = theme.useToken();
  const { initialState } = useModel('@@initialState');
  return <PageContainer></PageContainer>;
};

export default Welcome;
