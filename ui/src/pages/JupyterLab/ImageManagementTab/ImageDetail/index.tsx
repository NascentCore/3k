import React from 'react';
import { Button, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';

const Welcome: React.FC = () => {
  const intl = useIntl();

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '镜像名称',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'tag',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '镜像大小',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 100,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '推送时间',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '操作',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
            render: (_, record) => (
              <>
                <Space>
                  <Button type={'link'}>复制镜像地址</Button>
                  <Button type={'link'}>删除</Button>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={[{ xxx: 'xxx' }]}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Welcome;
