import { apiGetPayBalance, apiGetPayBilling } from '@/services';
import { useIntl, useModel } from '@umijs/max';
import { Button, Drawer, Tabs } from 'antd';
import React, { useEffect, useState } from 'react';
import DepositHistory from './DepositHistory';
import BillHistory from './BillHistory';

const Content = () => {
  const intl = useIntl();
  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.myAccount.tabs.title.DepositHistory',
        defaultMessage: '充值记录',
      }),
      children: <DepositHistory />,
    },
    {
      key: '2',
      label: intl.formatMessage({
        id: 'pages.myAccount.tabs.title.BillHistory',
        defaultMessage: '历史账单',
      }),
      children: <BillHistory />,
    },
  ];
  return (
    <>
      <Tabs defaultActiveKey="1" items={items} />
    </>
  );
};

const Index: React.FC = () => {
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const [balance, setBalance] = useState(0);
  const [show, setShow] = useState(false);
  useEffect(() => {
    apiGetPayBalance({ params: { user_id: currentUser?.user_id } }).then((res) => {
      setBalance(res?.balance);
      setShow(true);
    });
  }, []);

  return (
    <>
      {show && (
        <Button
          type="link"
          onClick={() => {
            setOpen(true);
          }}
        >
          {intl.formatMessage({
            id: 'pages.myAccount.button.balance',
            defaultMessage: '余额',
          })}
          : ￥{balance}
        </Button>
      )}
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'xxx',
          defaultMessage: intl.formatMessage({
            id: 'pages.myAccount.drawer.title.Account',
            defaultMessage: '账单信息',
          }),
        })}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && <Content />}
      </Drawer>
    </>
  );
};

export default Index;
