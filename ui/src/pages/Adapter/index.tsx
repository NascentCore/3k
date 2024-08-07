import { PageContainer } from '@ant-design/pro-components';
import { Button, Col, Input, Row, Table, Tabs } from 'antd';
import React, { useState } from 'react';
import { useApiResourceAdapters } from '@/services';
import { formatFileSize, removeUserIdPrefixFromPath } from '@/utils';
import { useIntl } from '@umijs/max';
const { Search } = Input;

const TabTable = ({ dataSource, loading }: any) => {
  const intl = useIntl();
  return (
    <Table
      columns={[
        {
          title: intl.formatMessage({
            id: 'pages.adapter.table.column.name',
            defaultMessage: '适配器名称',
          }),
          dataIndex: 'name',
          key: 'name',
          align: 'center',
          width: 300,
          sorter: (a: any, b: any) => {
            return a.name.toLowerCase().localeCompare(b.name);
          },
          render: (_) => removeUserIdPrefixFromPath(_),
        },
        {
          title: intl.formatMessage({
            id: 'pages.adapter.table.column.size',
            defaultMessage: '适配器体积',
          }),
          dataIndex: 'size',
          key: 'size',
          align: 'center',
          width: 150,
          render: (text) => {
            return formatFileSize(text);
          },
        },
      ]}
      dataSource={dataSource}
      loading={loading}
      scroll={{ y: 'calc(100vh - 350px)' }}
    />
  );
};

const Index: React.FC = () => {
  const intl = useIntl();
  const [searchText, setSearchText] = useState('');
  const { data, isLoading }: any = useApiResourceAdapters();
  console.log(1111, { data });
  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.adapter.tabs.title.public',
        defaultMessage: '公共适配器',
      }),
      children: (
        <>
          <TabTable
            dataSource={data?.public_list?.filter((x) => x?.name?.includes(searchText)) || []}
            loading={isLoading}
          />
        </>
      ),
    },
    {
      key: '2',
      label: intl.formatMessage({
        id: 'pages.adapter.tabs.title.user',
        defaultMessage: '私有适配器',
      }),
      children: (
        <>
          <TabTable
            dataSource={data?.user_list?.filter((x) => x?.name?.includes(searchText)) || []}
            loading={isLoading}
          />
        </>
      ),
    },
  ];
  return (
    <PageContainer>
      <Tabs
        defaultActiveKey="1"
        items={items}
        tabBarExtraContent={
          <>
            <Row gutter={10}>
              <Col>
                <Button
                  type="primary"
                  onClick={() => {
                    window.open(
                      'https://sxwl.ai/docs/document/cloud/sxwlctl-guide#%E4%B8%8A%E4%BC%A0',
                    );
                  }}
                >
                  {intl.formatMessage({
                    id: 'pages.global.button.upload',
                    defaultMessage: '上传',
                  })}
                </Button>
              </Col>
              <Col>
                <Search
                  allowClear
                  onInput={(e: any) => setSearchText(e.target.value || '')}
                  onSearch={(v) => setSearchText(v)}
                  style={{ width: 200 }}
                />
              </Col>
            </Row>
          </>
        }
      />
    </PageContainer>
  );
};

export default Index;
