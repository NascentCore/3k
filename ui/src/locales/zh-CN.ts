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
import global from './zh-CN/global';
import myAccount from './zh-CN/myAccount';
import clusterCpods from './zh-CN/clusterCpods';
import applicationMenu from './zh-CN/applicationMenu'

export default {
  'menu.welcome': '欢迎',
  'menu.ModelRepository': 'LLM 仓库',
  'menu.UserJobCommit': 'GPU 任务',
  'menu.UserJob': '任务详情',
  'menu.InferenceState': '推理实例',
  'menu.Grafana': '资源看板(Grafana)',
  'menu.Tensorboard': '训练看板(TensorBoard)',
  'menu.Jupyterlalb': '开发实验室(JupyterLab)',
  'menu.JupyterLab': '开发实验室',
  'menu.ClusterInformation': '集群信息',
  'menu.UserQuota': '用户限额',
  'menu.Dataset': '数据集',
  'menu.Adapter': 'LLM 适配器',
  'menu.ClusterCpods': '集群列表',
  'menu.ApplicationManagement':'大模型应用管理',
  'menu.ApplicationMenu': 'LLM 应用',
  'nav.title.Price': '价格',
  'nav.title.Document': '文档',
  ...global,
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
  ...myAccount,
  ...clusterCpods,
  ...applicationMenu
};
