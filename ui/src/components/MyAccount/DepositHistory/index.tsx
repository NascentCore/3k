import { useApiGetPayRecharge } from '@/services';
import { useIntl, useModel } from '@umijs/max';
import { Table } from 'antd';
import React from 'react';

const Index: React.FC = () => {
  const intl = useIntl();
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const { data } = useApiGetPayRecharge({
    params: { user_id: currentUser?.user_id, page: 1, page_size: 10000 },
  });
  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.DepositHistory.table.column.created_at',
              defaultMessage: '充值时间',
            }),
            dataIndex: 'created_at',
            key: 'created_at',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.DepositHistory.table.column.description',
              defaultMessage: '充值备注',
            }),
            dataIndex: 'description',
            key: 'description',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.DepositHistory.table.column.amount',
              defaultMessage: '充值金额',
            }),
            dataIndex: 'amount',
            key: 'amount',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.DepositHistory.table.column.after_balance',
              defaultMessage: '账户余额',
            }),
            dataIndex: 'after_balance',
            key: 'after_balance',
            align: 'center',
            width: 150,
          },
        ]}
        dataSource={data?.data || []}
        loading={false}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Index;
