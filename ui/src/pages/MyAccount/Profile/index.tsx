import { apiGetPayBalance } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Card, Descriptions } from 'antd';
import React, { useEffect } from 'react';

const Index: React.FC = () => {
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  console.log('currentUser', currentUser);
  useEffect(() => {
    apiGetPayBalance({ params: { user_id: currentUser?.id } });
  }, []);
  return (
    <PageContainer>
      <Card bordered={false}>
        <Descriptions title="基本信息" column={2}>
          <Descriptions.Item label="用户id">{currentUser?.id}</Descriptions.Item>
          <Descriptions.Item label="用户名">{currentUser?.username}</Descriptions.Item>
          <Descriptions.Item label="邮箱">{currentUser?.email}</Descriptions.Item>
          <Descriptions.Item label="companyId">{currentUser?.companyId}</Descriptions.Item>
          <Descriptions.Item label="isAdmin">{currentUser?.isAdmin}</Descriptions.Item>
          <Descriptions.Item label="userType">{currentUser?.userType}</Descriptions.Item>
          <Descriptions.Item label="创建时间">{currentUser?.createTime}</Descriptions.Item>
        </Descriptions>
      </Card>
    </PageContainer>
  );
};

export default Index;
