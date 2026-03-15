import { Button, Card, Col, Popconfirm, Row, Typography } from 'antd';
import React from 'react';
import { useIntl } from '@umijs/max';

const Index: React.FC = ({ record, onDelete, createAppJobAction }: any) => {
  const intl = useIntl();
  const isReference = record?.isReference === true;

  const isShowDeleteBtn =
    !isReference && (record?.jobItem?.status === 'running' || record?.jobItem?.status === 'failed');

  return (
    <Card
      title={record?.app_name}
      extra={
        <>
          {isShowDeleteBtn && (
            <Popconfirm
              title={intl.formatMessage({
                id: 'pages.global.confirm.title',
              })}
              onConfirm={() => onDelete(record)}
              okText={intl.formatMessage({
                id: 'pages.global.confirm.okText',
              })}
              cancelText={intl.formatMessage({
                id: 'pages.global.confirm.cancelText',
              })}
            >
              <Button type={'link'} danger>
                {intl.formatMessage({
                  id: 'pages.global.confirm.delete.button',
                })}
              </Button>
            </Popconfirm>
          )}
        </>
      }
      style={{ minWidth: 400, marginBottom: 10 }}
    >
      <Row>
        <Typography.Text>{record?.desc}</Typography.Text>
      </Row>
      <Row justify={'space-between'}>
        {isReference ? (
          record?.docUrl && (
            <Col>
              <Typography.Link href={record.docUrl} target="_blank" rel="noopener noreferrer">
                {intl.formatMessage({
                  id: 'pages.applicationMenu.appCard.integrationGuide',
                  defaultMessage: '查看集成指南',
                })}
              </Typography.Link>
            </Col>
          )
        ) : (
          <>
            {record?.jobItem?.status === 'running' && (
              <Col>
                {intl.formatMessage({
                  id: 'pages.applicationMenu.appCard.status',
                  defaultMessage: '状态',
                })}
                : {record?.jobItem?.status}
              </Col>
            )}
            {record?.jobItem?.status === 'running' && (
              <Col>
                <Typography.Link
                  onClick={() => {
                    window.open(record?.jobItem?.url);
                  }}
                >
                  {intl.formatMessage({
                    id: 'pages.applicationMenu.appCard.access',
                    defaultMessage: '访问知识库',
                  })}
                </Typography.Link>
              </Col>
            )}
            {!!record?.jobItem ||
              (record?.jobItem?.status !== 'running' && (
                <Col style={{ marginLeft: 'auto' }}>
                  <Typography.Link onClick={() => createAppJobAction(record)}>
                    {intl.formatMessage({
                      id: 'pages.applicationMenu.appCard.deployment',
                      defaultMessage: '部署',
                    })}
                  </Typography.Link>
                </Col>
              ))}
            {record?.jobItem && record?.jobItem?.status !== 'running' && (
              <Col style={{ marginLeft: 'auto' }}>
                <Typography.Link>
                  {intl.formatMessage({
                    id: 'pages.applicationMenu.appCard.onDeployment',
                    defaultMessage: '部署中',
                  })}
                </Typography.Link>
              </Col>
            )}
          </>
        )}
      </Row>
    </Card>
  );
};

export default Index;
