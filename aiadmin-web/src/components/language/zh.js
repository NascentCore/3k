module.exports = {
  message: {
    title: '提示',
    message: '当前登录状态已过期，请重新登录！',
    logout: '退出登录',
    logoutinfo: '确定退出吗？',
    logouttag: '提示',
    logoutsure: '确定',
    logoutcancel: '取消',
    switchlanguage: '切换语言成功',
    changelanguage: '切换语言',
    chinese: '中文',
    english: '英语',
    feedback: '问题反馈'
  },
  ai: {
    trainingsource: '训练数据源：',
    inputtrainingsource: '请输入训练数据源',
    mountpath: '挂载路径：',
    inputmountpath: '请输入挂载路径',
    ckptpath: 'CKPT 路径',
    inputckptpath: '请输入 CKPT 路径',
    ckptsavepath: '模型保存路径',
    inputckptsavepath: '请输入模型保存路径',
    capacity: '容量 ',
    capacityvol: 'MB',
    capacitymes: '请输入容量大小',
    gpumodel: '请选择型号',
    containername: '容器镜像',
    inputcontainername: '请输入容器镜像路径',
    jobtype: '任务类型：',
    typeinfo: '请选择类型',
    stopcondition: '终止条件',
    stopradio1: '自然终止',
    stopradio2: '设定时长：',
    hour: '分钟',
    submit: '提交',
    submiting: '任务提交中',
    valuenotnull: '值不能为空',
    ckpt_explain: '训练过程中产生的数据存放路径，对应用户训练程序中设置的路径',
    modepath_explain: '训练完成后的模型保存路径，对应用户训练程序中设置的保存路径',
    ckptvol_explain: 'CKPT 路径及模型保存路径将挂载对应的 PV ，容量是需要申请的 PV 大小，根据训练预估数据量大小填写',
    modevol_explain: '根据训练预估模型数据量大小，方便更好的存储模型',
    gpu_explain: '训练所需的 gpu 资源数量以及类型',
    container_explain: '用户需要将训练程序、训练数据以及所需环境打包成镜像，并将镜像上传到公网可访问的镜像仓库，镜像打包过程可参考附录一',
    jobtype_explain: '目前支持的任务类型为MPI',
    stop_explain: '可选择自然终止或手动设定运行时长，在设置运行时长到期后如任务未完成，该训练任务将被终止',
    priceper: '元/时/个'
  },
  login: {
    title: '算想云',
    username: '邮箱',
    usernameru: '不能为空',
    password: '密码',
    passwordru: '密码不能为空',
    rememberme: '记住我',
    login: '登 陆',
    logining: '登 陆 中...',
    registertitle: '邮箱注册',
    registername: '账户名',
    registeremail: '请输入邮箱',
    registercodemes: '邮箱验证码',
    registerpd: '密码',
    registerpd2: '确认密码',
    registerreadMe: '我同意算想未来隐私政策',
    registersub: '注册',
    registersubing: '注册中...',
    registerback: '返回',
    pdmes: '请再次输入密码',
    pdmesnosame: '两次输入密码不一致!',
    emailmes: '邮箱不能为空',
    emailmeserr: '邮箱格式错误',
    registersendcode: '获取验证码',
    registersendcode1: '验证码发送中',
    registersendcode2: '重新发送',
    namemes: '当前用户名不能为空',
    codemes: '验证码不能为空',
    pdmesnull: '当前密码不能为空',
    sendmesafter: '发送成功，验证码有效期5分钟',
    aftersec: '秒后重新发送',
    aftersecre: '秒后重新发送',
    powerregister: '算力源注册',
    userregister: '算力用户注册',
    email: '客服邮箱：',
    phone: '客服电话：'
  },
  router: {
    login: '登陆',
    tasksub: '任务提交',
    info: '任务详情',
    hashrate: '算力详情',
    guide: '使用指南',
    resultdownload: '下载模型',
    title: '',
    userguide: '使用导读',
    costcenter: '费用中心',
    orderinfo: '订单详情'
  },
  jobinfo: {
    jobName: '任务名称',
    beanName: '镜像名称',
    isPause: '运行状态',
    runstatus: '运行中',
    runstatusfail: '运行失败',
    runstatussuc: '运行成功',
    operate: '操作',
    monitor: '监控',
    checklog: '查日志',
    deleteinfo: '确定停止并删除该任务吗？',
    cancel: '取消',
    sure: '确定',
    delete: '删除',
    download: '下载训练结果',
    downloadlog: '下载日志',
    success: '成功',
    gpunumber: 'GPU数量',
    gputype: 'GPU型号',
    modelpath: 'Model保存路径',
    info: '详情',
    downloadpath: '模型下载路径如下：',
    suredel: '确定删除该任务吗？',
    ckptpath: 'CKPT 路径',
    trainingsource: '训练数据源',
    mountpath: '挂载路径',
    jobtype: '任务类型',
    createtime: '创建时间',
    payinfo: '支付宝扫码支付',
    goodname: '商品名称：',
    paytotal: '消费总计：',
    usedetail: '使用明细：',
    modelload: '模型加载中'
  },
  componyuser: {
    user: '算力源用户',
    name: '公司名称',
    accesskey: 'accesskey',
    componyphone: '公司联系方式',
    componyemail: '申请邮箱',
    createtime: '创建日期'
  },
  hashrate: {
    cpodid: 'Cpod ID',
    cpodv: 'Cpod 版本',
    gpuProd: 'GPU 型号',
    gpuvendor: 'GPU 厂商',
    gputotal: 'GPU 总数量',
    gpuAllocatable: 'GPU 可分配数量',
    createtime: '创建日期',
    updateTime: '更新时间'
  },
  order: {
    ordernum: '商户订单号',
    jobname: 'job名称',
    goodname: '商品名称',
    payamount: '支付金额',
    paystatus: '支付状态',
    nopay: '未支付',
    paid: '已支付',
    gooddetail: '商品明细',
    createtime: '创建日期',
    updatetime: '更新时间'
  }
}
