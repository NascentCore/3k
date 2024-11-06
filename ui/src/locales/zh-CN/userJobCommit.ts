/**
 * 任务提交国际化配置
 */
export default {
  'pages.UserJobCommit.form.placeholder': '请输入',
  'pages.UserJobCommit.form.ckptPath': '工作目录',
  'pages.UserJobCommit.form.ckptPath.tooltip':
    '训练过程中产生的数据存放路径，对应用户训练程序中设置的路径',
  'pages.UserJobCommit.form.ckptVol': '容量',
  'pages.UserJobCommit.form.ckptVol.tooltip':
    'CKPT 路径及模型保存路径将挂载对应的 PV ，容量是需要申请的 PV 大小，根据训练预估数据量大小填写',
  'pages.UserJobCommit.form.modelPath': '输出目录',
  'pages.UserJobCommit.form.modelPath.tooltip':
    '训练完成后的模型保存路径，对应用户训练程序中设置的保存路径',
  'pages.UserJobCommit.form.modelVol': '容量',
  'pages.UserJobCommit.form.modelVol.tooltip': '根据训练预估模型数据量大小，方便更好的存储模型',
  'pages.UserJobCommit.form.imagePath': '容器镜像',
  'pages.UserJobCommit.form.imagePath.tooltip':
    '用户需要将训练程序、训练数据以及所需环境打包成镜像，并将镜像上传到公网可访问的镜像仓库，镜像打包过程可参考附录一',
  'pages.UserJobCommit.form.jobType': '任务类型',
  'pages.UserJobCommit.form.stopType': '终止条件',
  'pages.UserJobCommit.form.stopType.tooltip':
    '可选择自然终止或手动设定运行时长，在设置运行时长到期后如任务未完成，该训练任务将被终止',
  'pages.UserJobCommit.form.Voluntary': '自然终止',
  'pages.UserJobCommit.form.SetDuration': '设定时长',
  'pages.UserJobCommit.form.stopTime.unit': '分钟',
  'pages.UserJobCommit.form.submit': '提交',
  'pages.UserJobCommit.form.submit.success': '提交成功',
  'pages.UserJobCommit.form.pretrainedModelId': '基座模型',
  'pages.UserJobCommit.form.pretrainedModelPath': '挂载路径',
  'pages.UserJobCommit.form.datasetId': '数据集',
  'pages.UserJobCommit.form.datasetPath': '挂载路径',
  'pages.UserJobCommit.form.path.error':'路径必须以"/"开头',
  'pages.UserJobCommit.form.runCommand': '启动命令',
  'pages.UserJobCommit.form.nodeCount': '节点数量',
};
