import { PageContainer } from '@ant-design/pro-components';
import React from 'react';

const Admin: React.FC = () => {
  const url = `${window.location.protocol}//${window.location.hostname}:30003`;
  return (
    <iframe src={url} style={{ width: '100%', height: 'calc(100vh - 120px)', border: 'none' }} />
  );
};

export default Admin;
