/**
 * @name 微调
 * @description 微调
 */
import { apiPostQuota, apiPutQuota, useGetApiUser, useGpuTypeOptions } from '@/services';
import { Button, Drawer, Form, Input, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

const Content = ({ type, record, onCancel, onChange }) => {
  const intl = useIntl();

  const gpuTypeOptions = useGpuTypeOptions({});

  const [form] = Form.useForm();

  useEffect(() => {
    if (type === 'edit') {
      record && form.setFieldsValue(record);
    }
  }, [record]);

  const { data: userListData } = useGetApiUser();
  const userOptions = userListData?.data?.map((x) => ({
    ...x,
    label: x.user_name,
    value: x.user_id,
  }));

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log('Form values:', values);
      if (type === 'add') {
        return apiPostQuota({ data: { ...values, quota: Number(values.quota) } }).then(() => {
          onChange();
          message.success(
            intl.formatMessage({
              id: 'pages.userQuota.edit.form.success',
              defaultMessage: '操作成功',
            }),
          );
          onCancel();
        });
      }
      if (type === 'edit') {
        return apiPutQuota({
          data: { ...values, id: record.id, quota: Number(values.quota) },
        }).then(() => {
          onChange();
          message.success(
            intl.formatMessage({
              id: 'pages.userQuota.edit.form.success',
              defaultMessage: '操作成功',
            }),
          );
          onCancel();
        });
      }
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="user_id"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.name',
            defaultMessage: '用户',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={userOptions}
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.name',
              defaultMessage: '用户',
            })}
            disabled={type === 'edit'}
          />
        </Form.Item>
        <Form.Item
          name="resource"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.role',
            defaultMessage: '资源类型',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={gpuTypeOptions}
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.role',
              defaultMessage: '资源类型',
            })}
            disabled={type === 'edit'}
          />
        </Form.Item>

        <Form.Item
          name="quota"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.gpu_product',
            defaultMessage: '资源配额',
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
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.gpu_product',
              defaultMessage: '资源配额',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.userQuota.edit.form.confirm',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.userQuota.edit.form.cancel',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = ({ type, record, onChange }) => {
  const intl = useIntl();

  const [open, setOpen] = useState(false);

  const btnTitle =
    type === 'edit'
      ? intl.formatMessage({
          id: 'pages.userQuota.edit.form.title',
          defaultMessage: '修改配额',
        })
      : intl.formatMessage({
          id: 'pages.userQuota.add.form.title',
          defaultMessage: '添加配额',
        });

  return (
    <>
      <Button
        type={type === 'edit' ? 'link' : 'primary'}
        onClick={() => {
          setOpen(true);
        }}
      >
        {btnTitle}
      </Button>
      <Drawer
        width={1000}
        title={btnTitle}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && (
          <Content
            type={type}
            record={record}
            onCancel={() => setOpen(false)}
            onChange={onChange}
          />
        )}
      </Drawer>
    </>
  );
};

export default Index;
