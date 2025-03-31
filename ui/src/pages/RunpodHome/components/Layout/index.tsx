import { Layout } from 'antd';
import Header from './Header';
import Sidebar from './Sidebar';

const { Content } = Layout;

/**
 * 页面布局组件
 * 包含侧边栏、顶部导航和内容区域
 */
interface LayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sidebar />
      <Layout style={{ marginLeft: 250 }}>
        <Header />
        <Content style={{ 
          padding: '16px', 
          background: '#fff',
          marginTop: '48px',
          minHeight: 'calc(100vh - 48px)',
          overflow: 'auto'
        }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout; 