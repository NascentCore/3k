import { useIntl } from '@umijs/max';
import { Table } from 'antd';
import React from 'react';

const Index: React.FC = () => {
  const intl = useIntl();
  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'ID',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '充值时间',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '充值备注',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '充值金额',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '账户余额',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
        ]}
        dataSource={[]}
        loading={false}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Index;
