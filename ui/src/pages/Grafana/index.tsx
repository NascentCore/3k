import React, { useEffect } from 'react';
import { useModel } from '@umijs/max';

const isDemoBuild = process.env.REACT_APP_DEMO === 'true';

const Grafana: React.FC = () => {
  const { initialState } = useModel('@@initialState');
  const isDemoUser = initialState?.currentUser?.user_id === 'demo';

  useEffect(() => {
    const main = (document as any).querySelector('main.ant-layout-content');
    if (main) main.style.padding = 0;
  }, []);

  const showDemoDashboard = isDemoBuild || isDemoUser;
  if (showDemoDashboard) {
    return (
      <div style={{ width: '100%', height: 'calc(100vh - 48px)', overflow: 'auto', background: '#0d1117' }}>
        <img
          src="/demo-resource-dashboard.png"
          alt="General / Nvidia GPU Metrics"
          style={{ width: '100%', height: 'auto', display: 'block' }}
        />
      </div>
    );
  }

  const url = `${window.location.protocol}//${window.location.hostname}:30006/d/Oxed_c6Wz/nvidia-dcgm-exporter-dashboard?orgId=1`;
  return (
    <div>
      <iframe src={url} style={{ width: '100%', height: 'calc(100vh - 60px)', border: 'none' }} title="Grafana" />
    </div>
  );
};

export default Grafana;
