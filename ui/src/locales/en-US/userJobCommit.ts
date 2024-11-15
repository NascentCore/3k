/**
 * 任务提交国际化配置
 */
export default {
  'pages.UserJobCommit.form.placeholder': 'Please enter',
  'pages.UserJobCommit.form.ckptPath': 'Working Directory',
  'pages.UserJobCommit.form.ckptPath.tooltip':
    "The path where the data generated during training is stored, corresponding to the path set in the user's training program",
  'pages.UserJobCommit.form.ckptVol': 'Capacity',
  'pages.UserJobCommit.form.ckptVol.tooltip':
    'The CKPT path and model save path will mount the corresponding PV. The capacity is the size of PV that needs to be requested, based on the estimated data size of the training',
  'pages.UserJobCommit.form.modelPath': 'Output Directory',
  'pages.UserJobCommit.form.modelPath.tooltip':
    "The path to save the model after training is completed, corresponding to the save path set in the user's training program",
  'pages.UserJobCommit.form.modelVol': 'Capacity',
  'pages.UserJobCommit.form.modelVol.tooltip':
    'Based on the estimated model data size of the training, for better model storage',
  'pages.UserJobCommit.form.imagePath': 'Container Image',
  'pages.UserJobCommit.form.imagePath.tooltip':
    'Users need to package the training program, training data, and required environment into an image, and upload the image to a publicly accessible image repository. The process of packaging the image can refer to Appendix A',
  'pages.UserJobCommit.form.jobType': 'Job Type',
  'pages.UserJobCommit.form.stopType': 'Stop Condition',
  'pages.UserJobCommit.form.stopType.tooltip':
    'You can choose natural termination or manually set the runtime duration. If the task is not completed after the set runtime duration expires, the training task will be terminated',
  'pages.UserJobCommit.form.Voluntary': 'Voluntary',
  'pages.UserJobCommit.form.SetDuration': 'Set Duration',
  'pages.UserJobCommit.form.stopTime.unit': 'Minutes',
  'pages.UserJobCommit.form.submit': 'Submit',
  'pages.UserJobCommit.form.submit.success': 'success',
  'pages.UserJobCommit.form.pretrainedModelId': 'Pretrained Model',
  'pages.UserJobCommit.form.pretrainedModelPath': 'Mount Path',
  'pages.UserJobCommit.form.datasetId': 'Dataset',
  'pages.UserJobCommit.form.datasetPath': 'Mount Path',
  'pages.UserJobCommit.form.path.error':'Mount path must start with "/"',
  'pages.UserJobCommit.form.runCommand': 'Run Command',
  'pages.UserJobCommit.form.nodeCount': 'Node Count',
  'pages.UserJobCommit.form.cluster': 'Cluster',
};
