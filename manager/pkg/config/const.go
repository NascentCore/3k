package config

const (
	//cpod manager相关的配置
	CPOD_NAMESPACE               = "cpod"                     //CPod 所工作的NameSpace
	PORTAL_JOBTYPE_MPI           = "MPI"                      //Portal中MPI JobType的表示
	URLPATH_FETCH_JOB            = "/api/userJob/cpod_jobs"   //获取Job的Path
	URLPATH_UPLOAD_CPOD_STATUS   = "/api/userJob/cpod_status" //上传信息的Path
	CPOD_CREATED_PVC_ACCESS_MODE = "ReadWriteMany"            //创建PVC时所指定的Access Mode
	//Model Uploader 相关的配置
	MODELUPLOADER_PVC_MOUNT_PATH = "/data"                              //在Pod中的PVC挂载路径
	STATUS_NEEDS_UPLOAD_MODEL    = "Complete"                           //需要上传模型时MPIJob的状态
	UPLOAD_STARTED_FLAG_FILE     = "upload_started_flag_file"           //标识上传工作开始的文件的名称
	FILE_UPLOAD_LOG_FILE         = "files_uploaded_log"                 //记录上传工作进度的文件的名称
	OSS_ENDPOINT                 = "http://oss-cn-beijing.aliyuncs.com" //阿里云OSS Endpoint
	OSS_BUCKET                   = "sxwl-ai"                            //OSS Bucket
)
