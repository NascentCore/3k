import { PageContainer } from '@ant-design/pro-components';
import { Row, Divider } from 'antd';
import React from 'react';
import MainLayout from './components/Layout';
import FeatureCards from './components/FeatureCards';
import UsageSection from './components/UsageSection';
import ResourceSection from './components/ResourceSection';

/**
 * RunpodHome 页面
 * 包含功能卡片、使用情况和资源监控等模块
 */
const Index: React.FC = () => {
  return (
    <MainLayout>
      <PageContainer title={false}>
        <FeatureCards />
        <Divider />
        <Row gutter={24}>
          <UsageSection />
          <ResourceSection />
        </Row>
      </PageContainer>
    </MainLayout>
  );
};

export default Index;
