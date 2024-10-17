import { PageContainer } from '@ant-design/pro-components';
import { Drawer, Flex } from 'antd';
import React, { useEffect, useState } from 'react';
import { apiDeleteAppJob, apiGetAppJob, apiGetAppList } from '@/services';
import { useIntl } from '@umijs/max';
import AppCard from './AppCard';
import AddAppForm from './AddAppForm';

const Index: React.FC = () => {
  const intl = useIntl();
  const [addAppJobOpen, setAddAppJobOpen] = useState(false);
  const [addAppJobRecord, setAddAppJobRecord] = useState(void 0);

  const [appList, setAppList] = useState([]);
  const initData = () => {
    apiGetAppList().then((applistRes) => {
      apiGetAppJob().then((appJobList) => {
        const applist: any = [];
        for (const appItem of applistRes?.data || []) {
          const jobItem = appJobList?.data?.find((x) => x.app_id === appItem.app_id);
          appItem.jobItem = jobItem;
          applist.push(appItem);
        }
        console.log(applist);
        setAppList(applist);
      });
    });
  };
  useEffect(() => {
    initData();

    // 每隔30秒刷新一次数据
    const interval = setInterval(() => {
      initData();
    }, 30000);

    // 组件卸载时清除定时器
    return () => clearInterval(interval);
  }, []);

  const deleteAction = async (record: any) => {
    const params = {
      job_name: record?.jobItem?.job_name,
    };
    console.log('params', params);
    apiDeleteAppJob({
      data: params,
    }).then((res) => {
      console.log('apiDeleteAppJob', res);
      initData();
    });
  };
  return (
    <PageContainer>
      <Flex wrap={'wrap'} gap="small">
        {appList?.map((item: any) => (
          <>
            <AppCard
              record={item}
              onDelete={deleteAction}
              createAppJobAction={(record: any) => {
                console.log('部署', record);
                setAddAppJobRecord(record);
                setAddAppJobOpen(true);
              }}
            />
          </>
        ))}
      </Flex>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.applicationMenu.darwer.title.deployApp',
          defaultMessage: '部署应用',
        })}
        placement="right"
        onClose={() => setAddAppJobOpen(false)}
        open={addAppJobOpen}
      >
        <AddAppForm
          record={addAppJobRecord}
          onChange={() => {
            setAddAppJobOpen(false);
            initData();
          }}
          onCancel={() => {
            setAddAppJobOpen(false);
          }}
        />
      </Drawer>
    </PageContainer>
  );
};

export default Index;
