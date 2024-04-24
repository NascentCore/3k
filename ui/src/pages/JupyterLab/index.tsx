import { PageContainer } from '@ant-design/pro-components';
import React from 'react';
import { Button, Drawer, Tabs } from 'antd';
import { useIntl } from '@umijs/max';
import JupyterLabTab from './JupyterLabTab';
import ImageManagementTab from './ImageManagementTab';
import AddJupyterLab from './AddJupyterLab';

const Index: React.FC = () => {
  const intl = useIntl();

  const onChange = (key: string) => {
    console.log(key);
  };

  const [addJupterLabOpen, setAddJupyterLabOpen] = React.useState(false);

  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.jupyterLab.tab.title.jupyterLabExample',
        defaultMessage: 'JupyterLab',
      }),
      children: <JupyterLabTab />,
    },
    {
      key: '2',
      label: intl.formatMessage({
        id: 'pages.jupyterLab.tab.title.imageManagement',
        defaultMessage: '镜像管理',
      }),
      children: <ImageManagementTab />,
    },
  ];
  return (
    <>
      <PageContainer>
        <div style={{ position: 'relative' }}>
          <Button
            style={{ position: 'absolute', right: 0, top: 5, zIndex: 10 }}
            onClick={() => setAddJupyterLabOpen(true)}
          >
            {intl.formatMessage({
              id: 'pages.jupyterLab.tab.createJupyterLabInstanceButton',
              defaultMessage: '创建JupyterLab实例',
            })}
          </Button>
          <Tabs defaultActiveKey="1" items={items} onChange={onChange} />
        </div>
      </PageContainer>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.jupyterLab.tab.createJupyterLabInstanceButton',
          defaultMessage: '创建JupyterLab实例',
        })}
        placement="right"
        onClose={() => setAddJupyterLabOpen(false)}
        open={addJupterLabOpen}
      >
        {addJupterLabOpen && (
          <AddJupyterLab
            onChange={() => setAddJupyterLabOpen(false)}
            onCancel={() => setAddJupyterLabOpen(false)}
          />
        )}
      </Drawer>
    </>
  );
};

export default Index;
