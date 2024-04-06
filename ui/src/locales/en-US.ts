import login from './en-US/login';
import modelRepository from './en-US/modelRepository';
import userJob from './en-US/userJob';
import inferenceState from './en-US/inferenceState';
import userJobCommit from './en-US/userJobCommit';
import clusterInformation from './en-US/clusterInformation';
import userQuota from './en-US/userQuota'

export default {
  'app.title': 'NascentCore Cloud',
  'menu.welcome': 'Welcome',
  'menu.ModelRepository': 'Model Repository',
  'menu.UserJob': 'Job Details',
  'menu.UserJobCommit': ' Job Submission',
  'menu.InferenceState': 'Inference Service State',
  'menu.Grafana': 'Grafana',
  'menu.Tensorboard': 'TensorBoard',
  'menu.Jupyterlalb': 'JupyterLab',
  'menu.ClusterInformation': 'ClusterInformation',
  'menu.UserQuota': 'UserQuota',
  'pages.global.header.logout': 'Log Out',
  'pages.global.confirm.title': 'Prompt',
  'pages.global.confirm.okText': 'Yes',
  'pages.global.confirm.cancelText': 'No',
  'pages.global.confirm.delete.button': 'Delete',
  'pages.global.confirm.delete.description': 'Confirm deletion?',
  'pages.global.confirm.delete.success': 'Delete Success',
  ...login,
  ...modelRepository,
  ...userJob,
  ...inferenceState,
  ...userJobCommit,
  ...clusterInformation,
  ...userQuota

};
