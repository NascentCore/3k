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

export default {
  'menu.welcome': 'Welcome',
  'menu.ModelRepository': 'Model Repository',
  'menu.UserJob': 'Job Details',
  'menu.UserJobCommit': 'Job Queue',
  'menu.InferenceState': 'Inference Service',
  'menu.Grafana': 'Grafana',
  'menu.Tensorboard': 'TensorBoard',
  'menu.Jupyterlalb': 'JupyterLab',
  'menu.JupyterLab': 'JupyterLab',
  'menu.ClusterInformation': 'ClusterInformation',
  'menu.UserQuota': 'UserQuota',
  'menu.Dataset': 'Dataset',
  'menu.Adapter': 'Adapter',
  'menu.ClusterCpods': 'ClusterCpods',
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
};
