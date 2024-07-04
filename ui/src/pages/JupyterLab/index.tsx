import { PageContainer } from '@ant-design/pro-components';
import React from 'react';
import { Button, Drawer, Tabs } from 'antd';
import { useIntl } from '@umijs/max';
import JupyterLabTab from './JupyterLabTab';
import ImageManagementTab from './ImageManagementTab';
import AddJupyterLab from './AddJupyterLab';
import { useApiGetJobJupyterImage, useApiGetJobJupyterlab } from '@/services';

const Index: React.FC = () => {
  const intl = useIntl();

  const onChange = (key: string) => {
    console.log(key);
  };

  const [addJupterLabType, setAddJupyterLabType] = React.useState('add');
  const [addJupterLabRecord, setAddJupterLabRecord] = React.useState(void 0);
  const [addJupterLabOpen, setAddJupyterLabOpen] = React.useState(false);
  const {
    data: tableDataSourceRes_1,
    mutate: mutate_1,
    isLoading: isLoading_1,
  } = useApiGetJobJupyterlab();
  const {
    data: tableDataSourceRes_2,
    mutate: mutate_2,
    isLoading: isLoading_2,
  } = useApiGetJobJupyterImage();

  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.jupyterLab.tab.title.jupyterLabExample',
        defaultMessage: 'JupyterLab实例',
      }),
      children: (
        <JupyterLabTab
          tableDataSourceRes={tableDataSourceRes_1}
          mutate={mutate_1}
          isLoading={isLoading_1}
          editBtnOnClick={(recode: any) => {
            setAddJupyterLabType('edit');
            setAddJupterLabRecord(recode);
            setAddJupyterLabOpen(true);
          }}
        />
      ),
    },
    {
      key: '2',
      label: intl.formatMessage({
        id: 'pages.jupyterLab.tab.title.imageManagement',
        defaultMessage: '镜像管理',
      }),
      children: (
        <ImageManagementTab
          tableDataSourceRes={tableDataSourceRes_2}
          mutate={mutate_2}
          isLoading={isLoading_2}
        />
      ),
    },
  ];
  return (
    <>
      <PageContainer>
        <div style={{ position: 'relative' }}>
          <Button
            style={{ position: 'absolute', right: 0, top: 5, zIndex: 10 }}
            onClick={() => {
              setAddJupyterLabType('add');
              setAddJupterLabRecord(void 0);
              setAddJupyterLabOpen(true);
            }}
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
          id:
            addJupterLabType === 'add'
              ? 'pages.jupyterLab.tab.createJupyterLabInstanceButton'
              : 'pages.jupyterLab.tab.updateJupyterLabInstanceButton',
          defaultMessage: '创建JupyterLab实例',
        })}
        placement="right"
        onClose={() => setAddJupyterLabOpen(false)}
        open={addJupterLabOpen}
      >
        {addJupterLabOpen && (
          <AddJupyterLab
            addJupterLabType={addJupterLabType}
            addJupterLabRecord={addJupterLabRecord}
            onChange={() => {
              setAddJupyterLabOpen(false);
              mutate_1();
              mutate_2();
            }}
            onCancel={() => setAddJupyterLabOpen(false)}
          />
        )}
      </Drawer>
    </>
  );
};

export default Index;
