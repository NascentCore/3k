import { LogoutOutlined, SettingOutlined, UserOutlined } from '@ant-design/icons';
import { history, useModel } from '@umijs/max';
import { Button, message, Popover, Spin, Typography } from 'antd';
import { createStyles } from 'antd-style';
import { stringify } from 'querystring';
import type { MenuInfo } from 'rc-menu/lib/interface';
import React, { useCallback, useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import HeaderDropdown from '../HeaderDropdown';
import { removeToken } from '@/utils';
import { useIntl } from '@umijs/max';
import MyAccount from '../MyAccount';

export type GlobalHeaderRightProps = {
  menu?: boolean;
  children?: React.ReactNode;
};

export const AvatarName = () => {
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  return <span className="anticon">{currentUser?.username}</span>;
};

const useStyles = createStyles(({ token }) => {
  return {
    action: {
      display: 'flex',
      height: '48px',
      marginLeft: 'auto',
      overflow: 'hidden',
      alignItems: 'center',
      padding: '0 8px',
      cursor: 'pointer',
      borderRadius: token.borderRadius,
      '&:hover': {
        backgroundColor: token.colorBgTextHover,
      },
    },
  };
});

export const AvatarDropdown: React.FC<GlobalHeaderRightProps> = ({ menu, children }) => {
  const intl = useIntl();
  /**
   * 退出登录，并且将当前的 url 保存
   */
  const loginOut = async () => {
    removeToken();
    const { search, pathname } = window.location;
    const urlParams = new URL(window.location.href).searchParams;
    /** 此方法会跳转到 redirect 参数所在的位置 */
    const redirect = urlParams.get('redirect');
    // Note: There may be security issues, please note
    if (window.location.pathname !== '/user/login' && !redirect) {
      history.replace({
        pathname: '/user/login',
        search: stringify({
          redirect: pathname + search,
        }),
      });
    }
  };
  const { styles } = useStyles();

  const { initialState, setInitialState } = useModel('@@initialState');

  const onMenuClick = useCallback(
    (event: MenuInfo) => {
      const { key } = event;
      if (key === 'logout') {
        flushSync(() => {
          setInitialState((s) => ({ ...s, currentUser: undefined }));
        });
        loginOut();
        return;
      }
      // history.push(`/account/${key}`);
    },
    [setInitialState],
  );

  const loading = (
    <span className={styles.action}>
      <Spin
        size="small"
        style={{
          marginLeft: 8,
          marginRight: 8,
        }}
      />
    </span>
  );

  if (!initialState) {
    return loading;
  }

  const { currentUser } = initialState;

  if (!currentUser || !currentUser.username) {
    return loading;
  }

  const [token, setToken] = useState('');
  useEffect(() => {
    try {
      const sxwl_token = localStorage.getItem('sxwl_token');
      if (sxwl_token) {
        setToken(sxwl_token.split(' ')[1]);
      }
    } catch (error) {
      console.log(error);
    }
  }, []);

  const menuItems = [
    ...(menu
      ? [
          {
            key: 'center',
            icon: <UserOutlined />,
            label: '个人中心',
          },
          {
            key: 'settings',
            icon: <SettingOutlined />,
            label: '个人设置',
          },
          {
            type: 'divider' as const,
          },
        ]
      : []),
    ,
    {
      key: 'userDetail',
      icon: <UserOutlined />,
      label: (
        <>
          <Popover
            placement="left"
            content={
              <div>
                <Typography.Text>
                  <pre style={{ maxWidth: 400 }}>{token}</pre>
                </Typography.Text>
                <div>
                  <Button
                    onClick={() => {
                      if (!navigator.clipboard) {
                        return;
                      }
                      navigator.clipboard.writeText(token).then(function () {
                        message.success('Copy success');
                      });
                    }}
                  >
                    复制
                  </Button>
                </div>
              </div>
            }
            title="Api Token"
          >
            <div>
              {intl.formatMessage({
                id: 'pages.global.header.accountDetail',
                defaultMessage: '账户详情',
              })}
            </div>
          </Popover>
        </>
      ),
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: intl.formatMessage({
        id: 'pages.global.header.logout',
        defaultMessage: '退出登录',
      }),
    },
  ];

  return (
    <>
      <HeaderDropdown
        menu={{
          selectedKeys: [],
          onClick: onMenuClick,
          items: menuItems,
        }}
      >
        {children}
      </HeaderDropdown>
      <MyAccount />
    </>
  );
};
