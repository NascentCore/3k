import login from './en-US/login';
import modelRepository from './en-US/modelRepository';
import userJob from './en-US/userJob';
import inferenceState from './en-US/inferenceState';

export default {
  'app.title': 'NascentCore.AI Cloud',
  'menu.welcome': 'Welcome',
  'menu.ModelRepository': 'Model Repository',
  'menu.UserJob': 'Job Details',
  'menu.InferenceState': 'Inference Service State',
  'menu.Grafana': 'Grafana',
  'menu.Tensorboard': 'Tensorboard',
  'menu.Jupyterlalb': 'Jupyterlalb',

  'pages.global.confirm.title': 'Prompt',
  'pages.global.confirm.okText': 'Yes',
  'pages.global.confirm.cancelText': 'No',
  'pages.global.confirm.delete.button': 'Delete',
  'pages.global.confirm.delete.description': 'Confirm deletion?',
  ...login,
  ...modelRepository,
  ...userJob,
  ...inferenceState,
};
