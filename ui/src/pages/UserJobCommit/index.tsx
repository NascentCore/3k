import { apiDeleteUserJob, useApiGetUserJob } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Popconfirm, Space, Table, theme } from 'antd';
import React, { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';

const Welcome: React.FC = () => {
  const intl = useIntl();

  return <PageContainer>任务提交</PageContainer>;
};

export default Welcome;
