import login from './zh-CN/login';
import modelRepository from './zh-CN/modelRepository';
import userJob from './zh-CN/userJob';
import inferenceState from './zh-CN/inferenceState';
import userJobCommit from './zh-CN/userJobCommit';
import clusterInformation from './zh-CN/clusterInformation';
import userQuota from './zh-CN/userQuota';
import JupyterLab from './zh-CN/JupyterLab';
import oem from './zh-CN/oem';
import Adapter from './zh-CN/adapter';
import Dataseta from './zh-CN/dataset';

export default {
  'menu.welcome': '欢迎',
  'menu.ModelRepository': '模型仓库',
  'menu.UserJobCommit': '任务提交',
  'menu.UserJob': '任务详情',
  'menu.InferenceState': '推理服务状态',
  'menu.Grafana': '系统资源看板',
  'menu.Tensorboard': '训练看板(TensorBoard)',
  'menu.Jupyterlalb': '开发实验室(JupyterLab)',
  'menu.JupyterLab': '开发实验室(JupyterLab)',
  'menu.ClusterInformation': '集群信息',
  'menu.UserQuota': '用户配额',
  'menu.Dataset': '数据集',
  'menu.Adapter': '适配器',
  'pages.global.header.logout': '退出登录',
  'pages.global.confirm.title': '提示',
  'pages.global.confirm.okText': '是',
  'pages.global.confirm.cancelText': '否',
  'pages.global.confirm.delete.button': '删除',
  'pages.global.confirm.delete.description': '确认删除?',
  'pages.global.confirm.delete.success': '删除成功',
  'pages.global.form.submit.success': '操作成功',
  'pages.global.form.placeholder': '请输入',
  'pages.global.form.select.placeholder': '请选择',
  ...oem,
  ...login,
  ...modelRepository,
  ...userJob,
  ...inferenceState,
  ...userJobCommit,
  ...clusterInformation,
  ...userQuota,
  ...JupyterLab,
  ...Adapter,
  ...Dataseta,
};
