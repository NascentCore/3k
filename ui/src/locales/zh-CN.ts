import login from './zh-CN/login';
import modelRepository from './zh-CN/modelRepository';
import userJob from './zh-CN/userJob';
import inferenceState from './zh-CN/inferenceState';
import userJobCommit from './zh-CN/userJobCommit';
import clusterInformation from './zh-CN/clusterInformation';
import userQuota from './zh-CN/userQuota'

export default {
  'app.title': '算想云',
  'menu.welcome': '欢迎',
  'menu.ModelRepository': '模型仓库',
  'menu.UserJobCommit': '任务提交',
  'menu.UserJob': '任务详情',
  'menu.InferenceState': '推理服务状态',
  'menu.Grafana': '系统资源看板',
  'menu.Tensorboard': 'AI训练看板(TensorBoard)',
  'menu.Jupyterlalb': '开发实验室(JupyterLab)',
  'menu.JupyterLab': '开发实验室(JupyterLab)',
  'menu.ClusterInformation': '集群信息',
  'menu.UserQuota': '用户配额',
  'pages.global.header.logout': '退出登录',
  'pages.global.confirm.title': '提示',
  'pages.global.confirm.okText': '是',
  'pages.global.confirm.cancelText': '否',
  'pages.global.confirm.delete.button': '删除',
  'pages.global.confirm.delete.description': '确认删除?',
  'pages.global.confirm.delete.success': '删除成功',
  ...login,
  ...modelRepository,
  ...userJob,
  ...inferenceState,
  ...userJobCommit,
  ...clusterInformation,
  ...userQuota
};
