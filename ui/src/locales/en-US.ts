import login from './en-US/login';
import modelRepository from './en-US/modelRepository';
import userJob from './en-US/userJob';
import inferenceState from './en-US/inferenceState';
import userJobCommit from './en-US/userJobCommit';
import clusterInformation from './en-US/clusterInformation';
import userQuota from './en-US/userQuota';
import JupyterLab from './en-US/JupyterLab';
import oem from './en-US/oem';
import Adapter from './en-US/adapter';
import Dataseta from './en-US/dataset';
import global from './en-US/global';
import myAccount from './en-US/myAccount';
import clusterCpods from './en-US/clusterCpods';
import applicationMenu from './en-US/applicationMenu'

export default {
  'menu.welcome': 'Welcome',
  'menu.ModelRepository': 'Models',
  'menu.UserJob': 'Job Details',
  'menu.UserJobCommit': 'Job Queue',
  'menu.InferenceState': 'Inference Endpoints',
  'menu.Grafana': 'Grafana',
  'menu.Tensorboard': 'TensorBoard',
  'menu.Jupyterlalb': 'JupyterLab',
  'menu.JupyterLab': 'Labs',
  'menu.ClusterInformation': 'ClusterInformation',
  'menu.UserQuota': 'UserQuota',
  'menu.Dataset': 'Datasets',
  'menu.Adapter': 'Adapters',
  'menu.ClusterCpods': 'Clusters',
  'menu.ApplicationManagement': 'Large Model Application Management',
  'menu.ApplicationMenu': 'Large Model Application',
  'nav.title.Price': 'Price',
  'nav.title.Document': 'Document',
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
