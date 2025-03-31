import { Layout, Typography, Space } from 'antd';

const { Header: AntHeader } = Layout;
const { Text } = Typography;

/**
 * 顶部导航栏组件
 * 显示页面标题和右侧信息
 */
const Header: React.FC = () => {
  return (
    <AntHeader style={{ 
      background: '#fff', 
      padding: '0 16px', 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center',
      borderBottom: '1px solid #f0f0f0',
      height: '48px',
      position: 'fixed',
      top: 0,
      right: 0,
      left: 250,
      zIndex: 1
    }}>
      <Text strong style={{ fontSize: '14px' }}>Home</Text>
      <Space size={16}>
        <Text style={{ fontSize: '13px' }}>Docs</Text>
        <Text style={{ fontSize: '13px' }}>Referrals</Text>
        <Text style={{ fontSize: '13px' }}>$25.00</Text>
      </Space>
    </AntHeader>
  );
};

export default Header; 