package v1beta1

const (
	ModelStorageLabel = "cpod.cpod/modelstorage"
)

const (
	// CPodJobSourceLabel  represents the source of the cpodjob, e.g. sxwl、kubectl etc.
	CPodJobSourceLabel                       = "cpod.cpod/cpodjob-source"
	CPodUserIDLabel                          = "cpod.cpod/userID"
	CPodUserNamespaceLabel                   = "cpod.cpod/userNamespace"
	CPodModelstorageNameAnno                 = "cpod.cpod/modelstorageName"
	CPodModelstorageBaseNameAnno             = "cpod.cpod/baseModelstorageName"
	CPodModelstorageDefaultFinetuneGPUCount  = "cpod.cpod/modelstorageName-finetune-gpu-count"
	CPodModelstorageDefaultInferenceGPUCount = "cpod.cpod/modelstorageName-inference-gpu-count"
	CPODStorageCopyLable                     = "cpod.cpod/storage-copy"

	CPodJobSource = "sxwl"

	CPodPublicNamespace     = "public"
	CPodPublicStorageSuffix = "-public"

	CPodPreTrainModelSizeAnno         = "cpod.cpod/cpod-model-size"
	CPodPreTrainModelReadableNameAnno = "cpod.cpod/cpod-model-readable-name"
	CPodDatasetSizeAnno               = "cpod.cpod/cpod-dataset-size"
	CPodDatasetlReadableNameAnno      = "cpod.cpod/cpod-dataset-readable-name"
)

const (
	ModelStoragePrefix = "modelstorage://"
)

const (
	// cpod manager相关的配置
	CPOD_NAMESPACE                 = "cpod"             // CPod 所工作的NameSpace
	PORTAL_JOBTYPE_MPI             = "MPI"              // Portal中MPI JobType的表示
	PORTAL_JOBTYPE_PYTORCH         = "Pytorch"          // Portal中对于PytorchJob类型的表示
	PORTAL_JOBTYPE_GENERAL         = "GeneralJob"       // Portal中对于GeneralJob类型的表示
	PORTAL_JOBTYPE_TENSORFLOW      = "TensorFlow"       // Portal中对于TensorFlowJob类型的表示
	PORTAL_STOPTYPE_WITHLIMIT      = 1                  // 定时结束
	PORTAL_STOPTYPE_VOLUNTARY_STOP = 0                  // 自行停止
	URLPATH_FETCH_JOB              = "/api/cpod/jobs"   // 获取Job的Path
	URLPATH_UPLOAD_CPOD_STATUS     = "/api/cpod/status" // 上传信息的Path
	CPOD_CREATED_PVC_ACCESS_MODE   = "ReadWriteMany"    // 创建PVC时所指定的Access Mode
	OSS_ACCESS_KEY_ENV_NAME        = "AK"               // oss access key在ModelUploader所在容器的环境中的名称，同时代表其在K8S Secret中所对应的Key的名称
	OSS_ACCESS_SECRET_ENV_NAME     = "AS"               // oss access secret在ModelUploader所在容器的环境中的名称，同时代表其在K8S Secret中所对应的Key的名称
	K8S_SECRET_NAME_FOR_OSS        = "akas4oss"         // 记录K8S访问密码的K8S secret的名称（运行CpodManager之前必须已经创建好）
	K8S_CPOD_CM                    = "cpod-info"
	TIME_FORMAT_FOR_K8S_LABEL      = "2006-01-02_15-04-05MST" // 用来作为K8S Labels的Time Format
	// Model Uploader 相关的配置
	MODELUPLOADER_PVC_MOUNT_PATH = "/data"                    // 在Pod中的PVC挂载路径
	STATUS_NEEDS_UPLOAD_MODEL    = "Complete"                 // 需要上传模型时MPIJob的状态
	UPLOAD_STARTED_FLAG_FILE     = "upload_started_flag_file" // 标识上传工作开始的文件的名称
	FILE_UPLOAD_LOG_FILE         = "files_uploaded_log"       // 记录上传工作进度的文件的名称
	PACK_FILE_NAME               = "data.zip"
	OSS_ENDPOINT                 = "https://oss-cn-beijing.aliyuncs.com" // 阿里云OSS Endpoint
	PRESIGNED_URL_FILE           = "presigned_url_file"                  // 记录文件下载链接的文件名称
	MARKET_ACCESS_KEY            = "access_key"                          // 云市场 access key
	OSS_URL_EXPIRED_SECOND       = 2592000                               // oss presigned url 过期秒数，默认30天
	URLPATH_UPLOAD_URLS          = "/api/cpod/model/url"                 // 上报 presigned url 接口
	MODELUPLOADER_JOBNAME_PREFIX = "modeluploader-"
	K8S_LABEL_NV_GPU_PRODUCT     = "nvidia.com/gpu.product"
	K8S_LABEL_NV_GPU_PRESENT     = "nvidia.com/gpu.present"
)
