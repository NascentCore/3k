import { apiDeleteInference, apiGetInference, apiGetUserJob } from '@/services';
import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Button, Space, Table, message, theme } from 'antd';
import React, { useEffect, useState } from 'react';

const Welcome: React.FC = () => {
  const [inferenceList, setInferenceList] = useState([]);
  useEffect(() => {
    apiGetInference({}).then((res) => {
      setInferenceList(res?.data);
    });
  }, []);
  return (
    <PageContainer>
      <Table
        columns={[
          {
            title: '推理服务名称',
            dataIndex: 'service_name',
            key: 'service_name',
            align: 'center',
            width: 150,
          },
          {
            title: '模型名称',
            dataIndex: 'model_id',
            key: 'model_id',
            align: 'center',
            width: 150,
          },
          {
            title: '推理服务状态',
            dataIndex: 'status',
            key: 'status',
            align: 'center',
            width: 150,
          },

          {
            title: 'url',
            dataIndex: 'url',
            key: 'url',
            align: 'center',
            width: 150,
          },
          {
            title: '启动时间',
            dataIndex: 'start_time',
            key: 'start_time',
            align: 'center',
            width: 150,
          },
          {
            title: '终止时间',
            dataIndex: 'end_time',
            key: 'end_time',
            align: 'center',
            width: 150,
          },

          {
            title: '操作',
            fixed: 'right',
            dataIndex: 'action',
            key: 'action',
            width: 200,
            align: 'center',
            render: (_, record) => (
              <>
                <Space>
                  <Button
                    onClick={() => {
                      console.log('打开聊天页面');
                    }}
                  >
                    启动聊天
                  </Button>

                  <Button
                    onClick={() => {
                      apiDeleteInference({
                        params: {
                          service_name: record.service_name,
                        },
                      }).then((res) => {
                        message.success('操作成功');
                        apiGetInference({}).then((res) => {
                          setInferenceList(res?.data);
                        });
                      });
                    }}
                  >
                    终止
                  </Button>
                </Space>
              </>
            ),
          },
        ]}
        dataSource={inferenceList}
        scroll={{ y: 'calc(100vh - 350px)' }}
      />
    </PageContainer>
  );
};

export default Welcome;
