/**
 * @name 微调
 * @description 微调
 */
import { apiFinetunes, apiPostApiNode, apiResourceDatasets, useApiGetGpuType } from '@/services';
import { Button, Col, Drawer, Form, Input, Row, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

const Content = ({ onCancel }) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({});

  const [resourceDatasetsOption, setResourceDatasets] = useState([]);
  useEffect(() => {
    apiResourceDatasets({}).then((res) => {
      setResourceDatasets(res?.map((x) => ({ ...x, label: x.name, value: x.name })));
    });
  }, []);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log('Form values:', values);
      return apiPostApiNode({
        data: values,
      }).then(() => {
        message.success(
          intl.formatMessage({
            id: 'pages.clusterInformation.add.form.success',
            defaultMessage: '添加成功',
          }),
        );
        onCancel();
      });
    });
  };

  return (
    <>
      <Form
        form={form}
        initialValues={formValues}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        style={{ maxWidth: 600 }}
      >
        <Form.Item
          name="node_name"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.table.column.name',
            defaultMessage: '节点名称',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.table.column.name',
              defaultMessage: '节点名称',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="node_role"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.table.column.role',
            defaultMessage: '节点类型',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={[
              {
                label: 'worker',
                value: 'worker',
              },
            ]}
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.table.column.role',
              defaultMessage: '节点类型',
            })}
          />
        </Form.Item>

        <Form.Item
          name="node_ip"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.add.form.node_ip',
            defaultMessage: '节点 IP',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.add.form.node_ip',
              defaultMessage: '节点 IP',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item
          name="ssh_port"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.add.form.node_ip',
            defaultMessage: 'SSH 端口',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="number"
            min={1}
            max={65535}
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.add.form.node_ip',
              defaultMessage: 'SSH 端口',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item
          name="ssh_user"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.add.form.ssh_user',
            defaultMessage: 'SSH 用户名',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.add.form.ssh_user',
              defaultMessage: 'SSH 用户名',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item
          name="ssh_password"
          label={intl.formatMessage({
            id: 'pages.clusterInformation.add.form.ssh_password',
            defaultMessage: 'SSH 密码',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.clusterInformation.add.form.ssh_password',
              defaultMessage: 'SSH 密码',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.confirm',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.cancel',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = ({ type }) => {
  const intl = useIntl();

  const [open, setOpen] = useState(false);

  return (
    <>
      <Button
        type={type === 'edit' ? 'link' : 'primary'}
        onClick={() => {
          setOpen(true);
        }}
      >
        {intl.formatMessage({
          id: 'pages.clusterInformation.add.form.title',
          defaultMessage: '新增节点',
        })}
      </Button>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.clusterInformation.add.form.title',
          defaultMessage: '新增节点',
        })}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && <Content onCancel={() => setOpen(false)} />}
      </Drawer>
    </>
  );
};

export default Index;
