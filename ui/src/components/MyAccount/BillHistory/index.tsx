import { useApiGetPayBilling, useApiGetPayBillingTasks } from '@/services';
import { useIntl, useModel } from '@umijs/max';
import { Table } from 'antd';
import React, { useState } from 'react';

const Index: React.FC = () => {
  const intl = useIntl();
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const handleTableChange = (pagination: any) => {
    setPage(pagination.current);
    setPageSize(pagination.pageSize);
  };
  const { data, isLoading } = useApiGetPayBillingTasks({
    params: { user_id: currentUser?.user_id, page: page, page_size: pageSize },
  });
  return (
    <>
      <Table
        pagination={{
          current: page,
          pageSize,
          total: data?.total,
          showTotal: (total) => `共 ${total} 条`,
        }}
        columns={[
          // {
          //   title: intl.formatMessage({
          //     id: 'pages.myAccount.BillHistory.table.column.billing_id',
          //     defaultMessage: 'ID',
          //   }),
          //   dataIndex: 'billing_id',
          //   key: 'billing_id',
          //   align: 'center',
          //   width: 150,
          // },
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
              defaultMessage: '服务名称',
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
              id: 'pages.myAccount.BillHistory.table.column.begin_time',
              defaultMessage: '开始时间',
            }),
            dataIndex: 'begin_time',
            key: 'begin_time',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'pages.myAccount.BillHistory.table.column.end_time',
              defaultMessage: '结束时间',
            }),
            dataIndex: 'end_time',
            key: 'end_time',
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
        scroll={{ y: 'calc(100vh - 320px)' }}
      />
      <p style={{ textAlign: 'right' }}>目前仅支持查询最近一个月内的账单记录</p>
    </>
  );
};

export default Index;
