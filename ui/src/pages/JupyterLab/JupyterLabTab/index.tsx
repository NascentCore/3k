import React, { useEffect } from 'react';
import { Button, Drawer, Space, Table } from 'antd';
import { useIntl } from '@umijs/max';
import BuildingImage from './BuildingImage';
import { apiGetJobJupyterlab } from '@/services';

const Welcome: React.FC = () => {
  const intl = useIntl();
  const [buildingImageOpen, setBuildingImageOpen] = React.useState(false);

  useEffect(() => {
    apiGetJobJupyterlab();
  }, []);

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '实例名称',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'CPU',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'MEM',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'GPU',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '操作',
            }),
            dataIndex: 'xxx',
            key: 'xxx',
            align: 'center',
            width: 150,
            render: (_, record) => (
              <>
                <Space>
                  <Button type={'link'}>进入</Button>
                  <Button type={'link'} onClick={() => setBuildingImageOpen(true)}>
                    构建镜像
                  </Button>
                  <Button type={'link'}>删除</Button>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={[{ xxx: 'xxx' }]}
        scroll={{ y: 'calc(100vh - 100px)' }}
      />
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'xxx',
          defaultMessage: '构建镜像',
        })}
        placement="right"
        onClose={() => setBuildingImageOpen(false)}
        open={buildingImageOpen}
      >
        <BuildingImage
          onChange={() => setBuildingImageOpen(false)}
          onCancel={() => setBuildingImageOpen(false)}
        />
      </Drawer>
    </>
  );
};

export default Welcome;
