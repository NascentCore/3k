import { apiGetPayBilling } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import React, { useEffect } from 'react';

const Index: React.FC = () => {
  useEffect(() => {
    apiGetPayBilling({});
  }, []);
  return <PageContainer>消费历史</PageContainer>;
};

export default Index;
