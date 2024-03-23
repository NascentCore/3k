import login from './zh-CN/login';
import modelRepository from './zh-CN/modelRepository';
import userJob from './zh-CN/userJob';
import inferenceState from './zh-CN/inferenceState';

export default {
  'app.title': '算想云',
  'menu.welcome': '欢迎',
  'menu.ModelRepository': '模型仓库',
  'menu.UserJob': '任务详情',
  'menu.InferenceState': '推理服务状态',
  'menu.Grafana': 'Grafana',
  'menu.Tensorboard': 'TensorBoard',
  'menu.Jupyterlalb': 'JupyterLab',
  'pages.global.header.logout': '退出登录',
  'pages.global.confirm.title': '提示',
  'pages.global.confirm.okText': '是',
  'pages.global.confirm.cancelText': '否',
  'pages.global.confirm.delete.button': '删除',
  'pages.global.confirm.delete.description': '确认删除?',
  ...login,
  ...modelRepository,
  ...userJob,
  ...inferenceState,
};
