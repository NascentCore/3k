import {
  apiFinetunes,
  useGpuTypeOptions,
  useResourceDatasetsOptions,
  useApiClusterCpods,
} from '@/services';
import { Button, Checkbox, Col, Drawer, Form, Input, Row, Select, message } from 'antd';
import { useState, useEffect, useMemo } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

interface ContentProps {
  record: any;
  onCancel: () => void;
  clusterPodsOptions: any;
}

const Content = ({ record, onCancel, clusterPodsOptions }: ContentProps) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({
    model: record?.name,
    gpu_count: record?.finetune_gpu_count || 1,
    finetune_type: 'lora',
    cluster_pod: undefined,
    hyperparameters: {
      n_epochs: '3.0',
      batch_size: '4',
      learning_rate_multiplier: '5e-5',
    },
    model_saved_type: false, // 默认为false
    model_category: record?.category
  });

  const resourceDatasetsOption = useResourceDatasetsOptions();
  const gpuTypeOptions = useGpuTypeOptions();
  const selectedClusterPod = Form.useWatch('cluster_pod', form);

  // 根据选中的集群过滤 GPU 选项
  const filteredGpuOptions = (() => {
    // 如果没有选择集群，返回所有 GPU 选项
    if (!selectedClusterPod) {
      return gpuTypeOptions;
    }

    // 如果选择了集群但数据未加载，返回所有 GPU 选项
    if (!clusterPodsOptions?.data?.[selectedClusterPod]) {
      return gpuTypeOptions;
    }

    const clusterNodes = clusterPodsOptions.data[selectedClusterPod];
    // 收集该集群所有节点的 GPU 型号
    const availableGpuTypes = new Set();
    
    // clusterNodes 是数组，每个元素代表一个节点
    clusterNodes.forEach((node: any) => {
      if (node.gpu_prod) {
        availableGpuTypes.add(node.gpu_prod);
      }
    });

    // 返回该集群拥有的 GPU 型号与全局 GPU 选项的交集
    return gpuTypeOptions.filter(option => 
      availableGpuTypes.has(option.value)
    );
  })();

  const gpuProdValue = Form.useWatch('gpu_model', form);
  const finetuneTypeValue = Form.useWatch('finetune_type', form);

  useEffect(() => {
    if (finetuneTypeValue === 'pt') {
      form.setFieldsValue({ model_saved_type: true });
    } else {
      form.setFieldsValue({ model_saved_type: false });
    }
  }, [finetuneTypeValue]);

  // 格式化集群选项
  const formattedClusterOptions = useMemo(() => {
    if (!clusterPodsOptions?.data) {
      return [];
    }
    return Object.keys(clusterPodsOptions.data).map(key => ({
      label: key,
      value: key
    }));
  }, [clusterPodsOptions?.data]);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      // 当 finetune_type 为 pt 时，强制设置 model_saved_type 为 'full'
      if (values.finetune_type === 'pt') {
        values.model_saved_type = 'full';
      } else {
        values.model_saved_type = values.model_saved_type ? 'full' : 'lora';
      }
      setFormValues(values);
      console.log('Form values:', values);
      const currentModel = record;
      const currentDataSet = resourceDatasetsOption.find(
        (x) => x.value === values.training_file,
      );
      const params = {
        ...values,
        cpod_id: values.cluster_pod,
        gpu_count: Number(values.gpu_count),
        model_id: currentModel.id,
        model_name: currentModel.name,
        model_path: currentModel.path,
        model_size: currentModel.size,
        model_is_public: currentModel.is_public,
        model_template: currentModel.template,
        model_category: currentModel.category,
        model_meta: currentModel.meta,
        dataset_id: currentDataSet.id,
        dataset_name: currentDataSet.name,
        dataset_path: currentDataSet.path,
        dataset_size: currentDataSet.size,
        dataset_is_public: currentDataSet.is_public,
      };
      console.log('params', params);
      return apiFinetunes({
        data: params,
      }).then(() => {
        message.success(
          intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.submit.success',
          }),
        );
        onCancel();
        history.push('/jobdetail');
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
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.model',
          })}
        >
          <span>{formValues.model}</span>
        </Form.Item>
        <Form.Item
          name="training_file"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.training_file',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={resourceDatasetsOption}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.training_file.placeholder',
            })}
          />
        </Form.Item>

        <Form.Item
          name="cluster_pod"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.clusterPod',
            defaultMessage: '集群',
          })}
        >
          <Select
            allowClear
            loading={!clusterPodsOptions}
            options={formattedClusterOptions}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.clusterPod.placeholder',
              defaultMessage: '请选择集群',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpu_model"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
            defaultMessage: 'GPU型号',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={filteredGpuOptions}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuProd',
              defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpu_count"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
            defaultMessage: 'GPU数量',
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
            max={
              gpuProdValue
                ? gpuTypeOptions?.find((x) => x.gpuProd === gpuProdValue).gpuAllocatable
                : 1
            }
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable',
              defaultMessage: 'GPU数量',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="finetune_type"
          label={intl.formatMessage({
            id: 'pages.modelRepository.fineTuningDrawer.form.finetune_type',
            defaultMessage: '类型',
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
                label: 'lora',
                value: 'lora',
              },
              {
                label: 'pt',
                value: 'pt',
              },
            ]}
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.training_file.placeholder',
            })}
          />
        </Form.Item>
        {finetuneTypeValue === 'lora' && (
          <Form.Item
            name="model_saved_type"
            valuePropName="checked"
            label={<div></div>}
            colon={false}
          >
            <Checkbox>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.form.model_saved_type',
                defaultMessage: '微调后保存完整模型（默认保存Lora）',
              })}
            </Checkbox>
          </Form.Item>
        )}

        <Row style={{ marginBottom: 15 }}>
          <Col span={8} style={{ textAlign: 'right' }}>
            Hyperparameters
          </Col>
          <Col span={16}></Col>
        </Row>

        <Form.Item
          name={['hyperparameters', 'n_epochs']}
          label="epochs"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name={['hyperparameters', 'batch_size']}
          label="batch_size"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name={['hyperparameters', 'learning_rate_multiplier']}
          label="Learning rate"
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            placeholder={intl.formatMessage({
              id: 'pages.modelRepository.fineTuningDrawer.form.input.placeholder',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.title',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.modelRepository.fineTuningDrawer.cancel',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

interface IndexProps {
  record: {
    id: string | number;
    name: string;
    path: string;
    size: number;
    is_public: boolean;
    template: any;
    category: string;
    meta: any;
    finetune_gpu_count?: number;
  }
}

const Index = ({ record }: IndexProps) => {
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  const clusterPodsOptions = useApiClusterCpods();

  return (
    <>
      <Button
        type="link"
        onClick={() => {
          setOpen(true);
        }}
      >
        {intl.formatMessage({
          id: 'pages.modelRepository.fineTuningDrawer.title',
        })}
      </Button>
      <Drawer
        width={1000}
        title={intl.formatMessage({
          id: 'pages.modelRepository.fineTuningDrawer.title',
        })}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && (
          <Content 
            record={record} 
            onCancel={() => setOpen(false)} 
            clusterPodsOptions={clusterPodsOptions}
          />
        )}
      </Drawer>
    </>
  );
};

export default Index;
