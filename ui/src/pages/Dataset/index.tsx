import { PageContainer } from '@ant-design/pro-components';
import { Button, Col, Input, Row, Table, Tabs, Modal, Select, message } from 'antd';
import React, { useState } from 'react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useApiResourceDatasets, useResourceModelsOptions, apiPostUserJob } from '@/services';
import { formatFileSize, removeUserIdPrefixFromPath } from '@/utils';
import { useIntl } from '@umijs/max';
const { Search } = Input;

// 注册 json 语言
SyntaxHighlighter.registerLanguage('json', json);

const TabTable = ({ dataSource, loading }: any) => {
  const intl = useIntl();
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [evaluateVisible, setEvaluateVisible] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<string>();
  const modelOptions = useResourceModelsOptions();

  const getSamplesCount = (record: any) => {
    if (!record.meta) return '-';
    try {
      const metaObj = typeof record.meta === 'string' ? JSON.parse(record.meta) : record.meta;
      return metaObj.total || '-';
    } catch {
      return '-';
    }
  };

  const handlePreview = (record: any) => {
    try {
      const metaObj = typeof record.meta === 'string' ? JSON.parse(record.meta) : record.meta;
      const previewContent = JSON.parse(metaObj.preview || '[]');
      setPreviewData(previewContent);
      setPreviewVisible(true);
    } catch {
      setPreviewData([]);
      setPreviewVisible(true);
    }
  };

  const handleEvaluate = (record: any) => {
    setSelectedDataset(record);
    setEvaluateVisible(true);
  };

  const handleEvaluateConfirm = async () => {
    if (!selectedModel) {
      message.error('请选择模型');
      return;
    }

    const params = {
      ckptPath: '/workspace/ckpt',
      ckptVol: 100,
      created_model_path: '/workspace/saved_model',
      created_model_vol: 100,
      nodeCount: 1,
      gpuNumber: 0,
      gpuType: '',
      imagePath: 'sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/evaluate:v1',
      jobType: 'General',
      dataset_id: selectedDataset.id,
      dataset_path: '/mnt/datasets',
      dataset_name: selectedDataset.name,
      dataset_size: selectedDataset.size,
      dataset_is_public: selectedDataset.is_public,
      model_id: selectedModel
    };

    try {
      await apiPostUserJob({ data: params });
      message.success('评测任务创建成功');
      setEvaluateVisible(false);
    } catch (error) {
      message.error('评测任务创建失败');
    }
  };

  return (
    <>
      <Table
        columns={[
          {
            title: intl.formatMessage({
              id: 'pages.dataset.table.column.name',
              defaultMessage: '数据集名称',
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
              id: 'pages.dataset.table.column.size',
              defaultMessage: '数据集大小',
            }),
            dataIndex: 'size',
            key: 'size',
            align: 'center',
            width: 150,
            render: (text) => {
              return formatFileSize(text);
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.dataset.table.column.samples',
              defaultMessage: '样本总数',
            }),
            dataIndex: 'samples',
            key: 'samples',
            align: 'center',
            width: 150,
            render: (_, record) => getSamplesCount(record),
          },
          {
            title: intl.formatMessage({
              id: 'pages.dataset.table.column.eval',
              defaultMessage: '评测集',
            }),
            dataIndex: 'eval',
            key: 'eval',
            align: 'center',
            width: 100,
            render: (_, record) => {
              try {
                const metaObj = typeof record.meta === 'string' ? JSON.parse(record.meta) : record.meta;
                const isEval = metaObj.eval || false;
                return intl.locale === 'zh-CN' ? (isEval ? '是' : '否') : (isEval ? 'true' : 'false');
              } catch {
                return intl.locale === 'zh-CN' ? '否' : 'false';
              }
            },
          },
          {
            title: intl.formatMessage({
              id: 'pages.dataset.table.column.actions',
              defaultMessage: '操作',
            }),
            key: 'actions',
            align: 'center',
            width: 150,
            render: (_, record) => (
              <>
                <Button type="link" onClick={() => handlePreview(record)}>
                  {intl.formatMessage({
                    id: 'pages.dataset.table.action.preview',
                    defaultMessage: '预览',
                  })}
                </Button>
                {(() => {
                  try {
                    const metaObj = typeof record.meta === 'string' ? JSON.parse(record.meta) : record.meta;
                    const isEval = metaObj.eval || false;
                    if (isEval) {
                      return (
                        <Button type="link" onClick={() => handleEvaluate(record)}>
                          {intl.formatMessage({
                            id: 'pages.dataset.table.action.evaluate',
                            defaultMessage: '评测',
                          })}
                        </Button>
                      );
                    }
                  } catch {
                    return null;
                  }
                })()}
              </>
            ),
          },
        ]}
        dataSource={dataSource}
        loading={loading}
        scroll={{ y: 'calc(100vh - 350px)' }}
      />

      <Modal
        title={
          intl.formatMessage(
            { 
              id: 'pages.dataset.preview.title.with.count',
              defaultMessage: '数据集预览（前5条记录）'
            }
          )
        }
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={800}
      >
        <div style={{ maxHeight: '60vh', overflow: 'auto' }}>
          <SyntaxHighlighter 
            language="json" 
            style={vscDarkPlus}
            customStyle={{
              margin: 0,
              borderRadius: '8px',
            }}
            codeTagProps={{
              style: {
                lineHeight: '1',
              }
            }}
          >
            {JSON.stringify(previewData, null, 2)}
          </SyntaxHighlighter>
        </div>
      </Modal>

      <Modal
        title={intl.formatMessage({
          id: 'pages.dataset.evaluate.title',
          defaultMessage: '选择评测模型',
        })}
        open={evaluateVisible}
        onOk={handleEvaluateConfirm}
        onCancel={() => setEvaluateVisible(false)}
      >
        <Select
          style={{ width: '100%' }}
          placeholder={intl.formatMessage({
            id: 'pages.dataset.evaluate.select.placeholder',
            defaultMessage: '请选择模型',
          })}
          options={modelOptions}
          value={selectedModel}
          onChange={(value) => setSelectedModel(value)}
        />
      </Modal>
    </>
  );
};

const Index: React.FC = () => {
  const intl = useIntl();
  const [searchText, setSearchText] = useState('');
  const { data, isLoading }: any = useApiResourceDatasets();

  const items = [
    {
      key: '1',
      label: intl.formatMessage({
        id: 'pages.dataset.tabs.title.public',
        defaultMessage: '公共数据集',
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
        id: 'pages.dataset.tabs.title.user',
        defaultMessage: '私有数据集',
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
                      'https://www.sxwl.ai/docs/document/cloud/sxwlctl-guide#%E4%B8%8A%E4%BC%A0',
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
