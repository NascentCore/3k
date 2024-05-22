import { PageContainer } from '@ant-design/pro-components';
import React, { useEffect } from 'react';

const Admin: React.FC = () => {
  useEffect(() => {
    (document as any).querySelector('main.ant-layout-content').style.padding = 0;
  }, []);
  const url = `${window.location.protocol}//${window.location.hostname}:30003`;
  return (
    <iframe src={url} style={{ width: '100%', height: 'calc(100vh - 60px)', border: 'none' }} />
  );
};

export default Admin;
