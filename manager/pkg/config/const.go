package config

const (
	//cpod manager相关的配置
	CPOD_NAMESPACE                = "cpod"                     //CPod 所工作的NameSpace
	PORTAL_JOBTYPE_MPI            = "MPI"                      //Portal中MPI JobType的表示
	URLPATH_FETCH_JOB             = "/api/userJob/cpod_jobs"   //获取Job的Path
	URLPATH_UPLOAD_CPOD_STATUS    = "/api/userJob/cpod_status" //上传信息的Path
	CPOD_CREATED_PVC_ACCESS_MODE  = "ReadWriteMany"            //创建PVC时所指定的Access Mode
	OSS_ACCESS_KEY_ENV_NAME       = "AK"                       //oss access key在ModelUploader所在容器的环境中的名称，同时代表其在K8S Secret中所对应的Key的名称
	OSS_ACCESS_SECRET_ENV_NAME    = "AS"                       //oss access secret在ModelUploader所在容器的环境中的名称，同时代表其在K8S Secret中所对应的Key的名称
	K8S_SECRET_NAME_FOR_OSS       = "akas4oss"                 //记录K8S访问密码的K8S secret的名称（运行CpodManager之前必须已经创建好）
	K8S_SA_NAME_FOR_MODELUPLOADER = "sa-modeluploader"         //Modeluploader用来访问集群所使用的 Service Account Name（运行CpodManager之前必须已经创建好）
	TIME_FORMAT_FOR_K8S_LABEL     = "2006-01-02_15-04-05"      //用来作为K8S Labels的Time Format
	//Model Uploader 相关的配置
	MODELUPLOADER_PVC_MOUNT_PATH = "/data"                              //在Pod中的PVC挂载路径
	STATUS_NEEDS_UPLOAD_MODEL    = "Complete"                           //需要上传模型时MPIJob的状态
	UPLOAD_STARTED_FLAG_FILE     = "upload_started_flag_file"           //标识上传工作开始的文件的名称
	FILE_UPLOAD_LOG_FILE         = "files_uploaded_log"                 //记录上传工作进度的文件的名称
	OSS_ENDPOINT                 = "http://oss-cn-beijing.aliyuncs.com" //阿里云OSS Endpoint

)
