/**
 * 模型仓库 国际化配置
 */

export default {
  'pages.modelRepository.tab.title.public': '公共模型',
  'pages.modelRepository.tab.title.user': '私有模型',

  'pages.modelRepository.table.column.id': '模型名称',
  'pages.modelRepository.table.column.owner': '所有者',
  'pages.modelRepository.table.column.category':'类型',
  'pages.modelRepository.table.column.size': '模型体积',
  'pages.modelRepository.table.column.action': '操作',

  'pages.modelRepository.fineTuningDrawer.title': '微调',
  'pages.modelRepository.fineTuningDrawer.cancel': '取消',
  'pages.modelRepository.fineTuningDrawer.submit.success': '微调任务创建成功',
  'pages.modelRepository.fineTuningDrawer.form.model': '模型名称',
  'pages.modelRepository.fineTuningDrawer.form.category': '类型',
  'pages.modelRepository.fineTuningDrawer.form.training_file': '数据集',
  'pages.modelRepository.fineTuningDrawer.form.gpuProd': 'GPU型号',
  'pages.modelRepository.fineTuningDrawer.form.gpuAllocatable': 'GPU数量',
  'pages.modelRepository.fineTuningDrawer.form.adapter': '适配器',
  'pages.modelRepository.fineTuningDrawer.form.finetune_type': '类型',
  'pages.modelRepository.fineTuningDrawer.form.model_saved_type':
    '微调后保存完整模型（默认保存Lora）',
  'pages.modelRepository.fineTuningDrawer.form.training_file.placeholder': '数据集',
  'pages.modelRepository.fineTuningDrawer.form.input.placeholder': '请输入',
  'pages.modelRepository.fineTuningDrawer.form.clusterPod': '集群',
  'pages.modelRepository.fineTuningDrawer.form.clusterPod.placeholder': '请选择集群',

  'pages.modelRepository.InferenceDrawer.title': '推理',
  'pages.modelRepository.InferenceDrawer.submit': '部署',
  'pages.modelRepository.InferenceDrawer.submit.success': '部署任务创建成功',
  'pages.modelRepository.InferenceDrawer.form.model_name': '模型名称',
  'pages.modelRepository.InferenceDrawer.form.autoScaling': '自动扩容配置',
  'pages.modelRepository.InferenceDrawer.form.minInstances': '最小实例数',
  'pages.modelRepository.InferenceDrawer.form.minInstances.placeholder': '请输入最小实例数',
  'pages.modelRepository.InferenceDrawer.form.maxInstances': '最大实例数',
  'pages.modelRepository.InferenceDrawer.form.maxInstances.placeholder': '请输入最大实例数',
  'pages.modelRepository.InferenceDrawer.validation.required': '此字段为必填项',
  'pages.modelRepository.InferenceDrawer.validation.instancesError': '最小实例数不能大于最大实例数',
  'pages.modelRepository.InferenceDrawer.validation.noGpu': '无可用GPU',
  'pages.modelRepository.InferenceDrawer.validation.maxInstancesError': '最大实例数不能超过 {maxInstances}（可用GPU数量 {totalGpu} / 每实例所需GPU数量 {gpuPerInstance}）',
};
