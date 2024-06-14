import { useApiGetPayBilling } from '@/services';
import { useIntl, useModel } from '@umijs/max';
import { Table } from 'antd';
import React from 'react';

const Index: React.FC = () => {
  const intl = useIntl();
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const { data, isLoading } = useApiGetPayBilling({
    params: { user_id: currentUser?.user_id },
  });
  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.billing_id',
              defaultMessage: 'ID',
            }),
            dataIndex: 'billing_id',
            key: 'billing_id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.job_id',
              defaultMessage: '任务ID',
            }),
            dataIndex: 'job_id',
            key: 'job_id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.job_type',
              defaultMessage: '任务类型',
            }),
            dataIndex: 'job_type',
            key: 'job_type',
            align: 'center',
            width: 150,
          },
          // {
          //   title: intl.formatMessage({
          //     id: 'xxx',
          //     defaultMessage: 'GPU类型',
          //   }),
          //   dataIndex: 'xxx',
          //   key: 'xxx',
          //   align: 'center',
          //   width: 150,
          // },
          // {
          //   title: intl.formatMessage({
          //     id: 'xxx',
          //     defaultMessage: 'GPU数量',
          //   }),
          //   dataIndex: 'xxx',
          //   key: 'xxx',
          //   align: 'center',
          //   width: 150,
          // },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.billing_time',
              defaultMessage: '账单计费时间',
            }),
            dataIndex: 'billing_time',
            key: 'billing_time',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.payment_time',
              defaultMessage: '账单生成时间',
            }),
            dataIndex: 'payment_time',
            key: 'payment_time',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.amount',
              defaultMessage: '金额',
            }),
            dataIndex: 'amount',
            key: 'amount',
            align: 'center',
            width: 150,
          },
        ]}
        dataSource={data?.data || []}
        loading={isLoading}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
    </>
  );
};

export default Index;
