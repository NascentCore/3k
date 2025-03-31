import { Layout, Menu, Typography, Space, Avatar, Tag } from 'antd';
import React from 'react';
import {
  HomeOutlined,
  CompassOutlined,
  AppstoreOutlined,
  UserOutlined,
  QuestionCircleOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  KeyOutlined,
  ClusterOutlined,
  DesktopOutlined,
  SettingOutlined,
  CreditCardOutlined,
  WalletOutlined,
  TeamOutlined,
  AuditOutlined,
  DesktopOutlined as RemoteAccessOutlined,
  MessageOutlined,
  QuestionCircleOutlined as FAQOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;
const { Text } = Typography;

/**
 * 菜单项配置
 */
const menuItems = [
  {
    key: 'home',
    icon: <HomeOutlined />,
    label: 'Home',
  },
  {
    key: 'explore',
    icon: <CompassOutlined />,
    label: 'Explore',
  },
  {
    key: 'manage',
    icon: <AppstoreOutlined />,
    label: 'Manage',
    children: [
      {
        key: 'pods',
        icon: <CloudServerOutlined />,
        label: 'Pods',
      },
      {
        key: 'serverless',
        icon: <ThunderboltOutlined />,
        label: 'Serverless',
      },
      {
        key: 'fine-tuning',
        icon: <DatabaseOutlined />,
        label: (
          <Space>
            Fine Tuning
            <Tag color="blue" style={{ marginLeft: 4 }}>BETA</Tag>
          </Space>
        ),
      },
      {
        key: 'storage',
        icon: <DatabaseOutlined />,
        label: 'Storage',
      },
      {
        key: 'templates',
        icon: <FileTextOutlined />,
        label: 'Templates',
      },
      {
        key: 'secrets',
        icon: <KeyOutlined />,
        label: 'Secrets',
      },
      {
        key: 'instant-cluster',
        icon: <ClusterOutlined />,
        label: (
          <Space>
            Instant Cluster
            <Tag color="blue" style={{ marginLeft: 4 }}>BETA</Tag>
          </Space>
        ),
      },
      {
        key: 'bare-metal',
        icon: <DesktopOutlined />,
        label: (
          <Space>
            Bare Metal
            <Tag color="green" style={{ marginLeft: 4 }}>NEW</Tag>
          </Space>
        ),
      },
    ],
  },
  {
    key: 'account',
    icon: <UserOutlined />,
    label: 'Account',
    children: [
      {
        key: 'settings',
        icon: <SettingOutlined />,
        label: 'Settings',
      },
      {
        key: 'billing',
        icon: <CreditCardOutlined />,
        label: 'Billing',
      },
      {
        key: 'savings-plans',
        icon: <WalletOutlined />,
        label: 'Savings Plans',
      },
      {
        key: 'team',
        icon: <TeamOutlined />,
        label: 'Team',
      },
      {
        key: 'audit-logs',
        icon: <AuditOutlined />,
        label: 'Audit Logs',
      },
      {
        key: 'remote-access',
        icon: <RemoteAccessOutlined />,
        label: 'Remote Access',
      },
    ],
  },
  {
    key: 'help',
    icon: <QuestionCircleOutlined />,
    label: 'Help',
    children: [
      {
        key: 'contact',
        icon: <MessageOutlined />,
        label: 'Contact',
      },
      {
        key: 'faq',
        icon: <FAQOutlined />,
        label: 'FAQ',
      },
    ],
  },
];

/**
 * 默认展开的菜单项
 */
const defaultOpenKeys = ['manage', 'account', 'help'];

/**
 * 侧边栏组件
 * 包含用户信息和导航菜单
 */
const Sidebar: React.FC = () => {
  return (
    <Sider 
      width={250} 
      theme="light" 
      style={{ 
        borderRight: '1px solid #f0f0f0', 
        position: 'fixed', 
        left: 0, 
        top: 0, 
        bottom: 0,
        overflow: 'auto',
        height: '100vh'
      }}
    >
      {/* 用户信息区域 */}
      <div style={{ 
        padding: '12px 16px', 
        borderBottom: '1px solid #f0f0f0',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        height: '48px'
      }}>
        <Avatar size="small" icon={<UserOutlined />} />
        <Text strong style={{ fontSize: '14px' }}>userName</Text>
      </div>

      {/* 导航菜单 */}
      <Menu
        mode="inline"
        defaultSelectedKeys={['home']}
        defaultOpenKeys={defaultOpenKeys}
        style={{ 
          height: 'calc(100% - 48px)', 
          borderRight: 0,
          overflow: 'auto'
        }}
        items={menuItems}
      />
    </Sider>
  );
};

export default Sidebar; 