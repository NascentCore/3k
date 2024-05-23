import { PageContainer } from '@ant-design/pro-components';
import React, { useEffect } from 'react';

const Admin: React.FC = () => {
  useEffect(() => {
    (document as any).querySelector('main.ant-layout-content').style.padding = 0;
  }, []);
  const url = `${window.location.protocol}//${window.location.hostname}:30006/d/Oxed_c6Wz/nvidia-dcgm-exporter-dashboard?orgId=1`;
  return (
    <div>
      <iframe src={url} style={{ width: '100%', height: 'calc(100vh - 60px)', border: 'none' }} />
    </div>
  );
};

export default Admin;
