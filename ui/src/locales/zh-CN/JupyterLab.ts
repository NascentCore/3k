/**
 * 开发实验室 国际化
 */
export default {
  'pages.jupyterLab.tab.title.jupyterLabExample': '实例',
  'pages.jupyterLab.tab.title.imageManagement': '镜像管理',
  'pages.jupyterLab.tab.createJupyterLabInstanceButton': '创建JupyterLab实例',
  'pages.jupyterLab.JupyterLabTab.table.column.instance_name': '实例名称',
  'pages.jupyterLab.JupyterLabTab.table.column.cpu_count': 'CPU',
  'pages.jupyterLab.JupyterLabTab.table.column.memory': 'MEM',
  'pages.jupyterLab.JupyterLabTab.table.column.gpu_product': 'GPU',
  'pages.jupyterLab.JupyterLabTab.table.column.status': '状态',
  'pages.jupyterLab.JupyterLabTab.table.column.action': '操作',
  'pages.jupyterLab.JupyterLabTab.table.column.action.enterBtn': '进入',
  'pages.jupyterLab.JupyterLabTab.table.column.action.buildBtn': '构建镜像',
  'pages.jupyterLab.JupyterLabTab.BuildingImage.form.base_image': '基础镜像',
  'pages.jupyterLab.JupyterLabTab.BuildingImage.form.tips':
    '构建镜像时会执行如下逻辑：<br/>1. 将 /workspace 目录下的内容完整复制到镜像相同路径下<br/>2. 自动安装 /workspace 目录下的 requirements.txt<br/>请将代码及 requirements.txt 文件放到该路径下<br/>注：数据卷默认挂载路径为 /workspace',
  'pages.jupyterLab.AddJupyterLab.form.instance_name': '实例名称',
  'pages.jupyterLab.AddJupyterLab.form.instance_name.pattern':
    '实例名称只能包含大小写字母、数字、下划线(_)和分割符(-)',
  'pages.jupyterLab.AddJupyterLab.form.cpu_count': 'CPU',
  'pages.jupyterLab.AddJupyterLab.form.memory': 'MEM',
  'pages.jupyterLab.AddJupyterLab.form.gpu_count': 'GPU数量',
  'pages.jupyterLab.AddJupyterLab.form.gpu_product': 'GPU类型',
  'pages.jupyterLab.AddJupyterLab.form.data_volume_size': '数据卷大小',
  'pages.jupyterLab.AddJupyterLab.form.model_id': '挂载模型',
  'pages.jupyterLab.AddJupyterLab.form.model_path': '模型挂载路径',
  'pages.jupyterLab.AddJupyterLab.form.datasets': '挂载数据集',
  'pages.jupyterLab.AddJupyterLab.form.adapters': '挂载适配器',
  'pages.jupyterLab.ImageManagementTab.table.image_name': '镜像名称',
  'pages.jupyterLab.ImageManagementTab.table.create_time': '创建时间',
  'pages.jupyterLab.ImageManagementTab.table.push_time': '更新时间',
  'pages.jupyterLab.ImageManagementTab.table.action': '操作',
  'pages.jupyterLab.ImageManagementTab.table.action.detail': '详情',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.image_name': '镜像名称',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.tag': 'Tag',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.image_size': '镜像大小',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.push_time': '推送时间',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action': '操作',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action.copy': '复制镜像地址',
  'pages.jupyterLab.ImageManagementTab.ImageDetail.table.action.delete': '删除',
};
