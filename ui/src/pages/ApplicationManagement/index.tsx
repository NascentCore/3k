import { PageContainer } from '@ant-design/pro-components';
import {
  Button,
  Card,
  Col,
  Drawer,
  Flex,
  Input,
  Popconfirm,
  Row,
  Space,
  Table,
  Tabs,
  Typography,
} from 'antd';
import React, { useEffect, useState } from 'react';
import {
  apiDeleteAppJob,
  apiDeleteAppList,
  apiGetAppJob,
  apiGetAppList,
  useApiResourceAdapters,
} from '@/services';
import { formatFileSize, removeUserIdPrefixFromPath } from '@/utils';
import { useIntl } from '@umijs/max';
import {
  ArrowDownOutlined,
  DeleteOutlined,
  EditOutlined,
  EllipsisOutlined,
  FormOutlined,
  MinusOutlined,
  PlusOutlined,
  SettingOutlined,
  StopOutlined,
} from '@ant-design/icons';
import AddAppForm from './AddAppForm';
const { Search } = Input;

const Index: React.FC = () => {
  const intl = useIntl();
  const [addAppOpen, setAddAppOpen] = useState(false);
  const [addAppRecord, setAddAppRecord] = useState(void 0);
  const [dataSource, setDataSource] = useState([]);

  useEffect(() => {
    apiGetAppList().then((res) => {
      setDataSource(res.data || []);
    });
  }, []);

  const deleteAction = async (record: any) => {
    // apiDeleteAppList({
    //   data: record,
    // }).then((res) => {
    //   apiGetAppList().then((res) => {
    //     setDataSource(res.data || []);
    //   });
    // });
  };
  return (
    <PageContainer>
      <div style={{ marginBottom: 20, textAlign: 'right' }}>
        <Button
          type={'primary'}
          icon={<PlusOutlined />}
          onClick={() => {
            setAddAppRecord(void 0);
            setAddAppOpen(true);
          }}
        >
          添加应用
        </Button>
      </div>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: 'id',
            }),
            dataIndex: 'id',
            key: 'id',
            align: 'center',
            width: 300,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '名称',
            }),
            dataIndex: 'name',
            key: 'name',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '描述',
            }),
            dataIndex: 'desc',
            key: 'desc',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '状态',
            }),
            dataIndex: 'status',
            key: 'status',
            align: 'center',
            width: 150,
            render: (_, record) => {
              return <>{_ === 1 ? '已上架' : '已下架'}</>;
            },
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '创建时间',
            }),
            dataIndex: 'created_at',
            key: 'created_at',
            align: 'center',
            width: 150,
          },

          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '创建人',
            }),

            dataIndex: 'user_id',
            key: 'user_id',
            align: 'center',
            width: 150,
          },
          {
            title: intl.formatMessage({
              id: 'xxx',
              defaultMessage: '操作',
            }),
            fixed: 'right',
            dataIndex: 'action',
            key: 'action',
            align: 'center',
            width: 300,
            render: (_, record) => {
              return (
                <>
                  <Space>
                    {/* 下架 */}
                    <Popconfirm
                      title={intl.formatMessage({
                        id: 'pages.global.confirm.title',
                      })}
                      onConfirm={() => {}}
                      okText={intl.formatMessage({
                        id: 'pages.global.confirm.okText',
                      })}
                      cancelText={intl.formatMessage({
                        id: 'pages.global.confirm.cancelText',
                      })}
                    >
                      <Button type={'link'} icon={<ArrowDownOutlined />}>
                        下架
                      </Button>
                    </Popconfirm>
                    {/* 编辑 */}
                    <Button
                      type={'link'}
                      icon={<FormOutlined />}
                      onClick={() => {
                        setAddAppRecord(record);
                        setAddAppOpen(true);
                      }}
                    >
                      编辑
                    </Button>
                    {/* 删除 */}
                    <Popconfirm
                      title={intl.formatMessage({
                        id: 'pages.global.confirm.title',
                      })}
                      onConfirm={() => deleteAction(record)}
                      okText={intl.formatMessage({
                        id: 'pages.global.confirm.okText',
                      })}
                      cancelText={intl.formatMessage({
                        id: 'pages.global.confirm.cancelText',
                      })}
                    >
                      <Button type={'link'} icon={<DeleteOutlined />}>
                        删除
                      </Button>
                    </Popconfirm>
                  </Space>
                </>
              );
            },
          },
        ]}
        dataSource={dataSource}
        scroll={{ y: 'calc(100vh - 350px)' }}
      />
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'xxx',
          defaultMessage: '编辑',
        })}
        placement="right"
        onClose={() => setAddAppOpen(false)}
        open={addAppOpen}
      >
        {addAppOpen && (
          <AddAppForm
            record={addAppRecord}
            onChange={() => {
              setAddAppOpen(false);
              apiGetAppList().then((res) => {
                setDataSource(res.data || []);
              });
            }}
            onCancel={() => {
              setAddAppOpen(false);
            }}
          />
        )}
      </Drawer>
    </PageContainer>
  );
};

export default Index;
