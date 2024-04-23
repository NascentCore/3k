import React, { useState } from 'react';
import { Button, Drawer, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import ImageDetail from './ImageDetail';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [detailDrawerOpen, setDetailDrawerOpen] = useState(false);
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
              defaultMessage: '创建时间',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '更新时间',
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
                  <Button type={'link'} onClick={() => setDetailDrawerOpen(true)}>
                    详情
                  </Button>
                  <Button type={'link'}>删除</Button>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={[{ xxx: 'xxx' }]}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'xxx',
          defaultMessage: '镜像详情',
        })}
        placement="right"
        onClose={() => setDetailDrawerOpen(false)}
        open={detailDrawerOpen}
      >
        <ImageDetail />
      </Drawer>
    </>
  );
};

export default Welcome;
