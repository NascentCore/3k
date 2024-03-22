import { PageContainer } from '@ant-design/pro-components';
import React from 'react';

const Admin: React.FC = () => {
  const url = `${window.location.protocol}//${window.location.hostname}:300006`;
  return (
    <PageContainer>
      <iframe src={url} style={{ width: '100%', height: 'calc(100vh - 170px)', border: 'none' }} />
    </PageContainer>
  );
};

export default Admin;
