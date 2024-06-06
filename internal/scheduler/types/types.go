// Code generated by goctl. DO NOT EDIT.
package types

type BalanceAddReq struct {
	UserID string  `header:"Sx-User-ID"`
	ToUser string  `json:"user_id"`
	Amount float64 `json:"amount"`
}

type BalanceAddResp struct {
	Message string `json:"message"`
}

type BalanceGetReq struct {
	UserID string `header:"Sx-User-ID"`
	ToUser string `form:"user_id"`
}

type BalanceGetResp struct {
	UserID  string  `json:"user_id"`
	Balance float64 `json:"balance"`
}

type BaseImageListReq struct {
	UserID string `header:"Sx-User-ID"`
}

type BaseImageListResp struct {
	Data []string `json:"data"`
}

type BaseReq struct {
	UserID string `header:"Sx-User-ID"`
}

type BaseResp struct {
	Message string `json:"message"`
}

type BillingListReq struct {
	UserID    string `header:"Sx-User-ID"`
	ToUser    string `form:"user_id,optional"`
	StartTime string `form:"start_time,optional"`
	EndTime   string `form:"end_time,optional"`
	JobID     string `form:"job_id,optional"`
}

type BillingListResp struct {
	Data []UserBilling `json:"data"`
}

type BuildImageReq struct {
	BaseImage    string `json:"base_image"`
	UserID       string `json:"user_id"`
	InstanceName string `json:"instance_name"`
	JobName      string `json:"job_name"`
}

type CPODStatusReq struct {
	JobStatus        []JobStatus        `json:"job_status"`
	InferenceStatus  []InferenceStatus  `json:"inference_status"`
	JupyterlabStatus []JupyterlabStatus `json:"jupyter_status"`
	ResourceInfo     ResourceInfo       `json:"resource_info"`
	UpdateTime       string             `json:"update_time"`
	CPODID           string             `json:"cpod_id"`
	UserID           string             `header:"Sx-User-ID"`
}

type CPODStatusResp struct {
	Message string `json:"message"`
}

type CPUInfo struct {
	Cores int `json:"cores"`
	Usage int `json:"usage"`
}

type Cache struct {
	DataType          string `json:"data_type"`
	DataName          string `json:"data_name"`
	DataId            string `json:"data_id"`
	DataSize          int64  `json:"data_size"`
	Template          string `json:"template"`
	DataSource        string `json:"data_source"`
	IsPublic          bool   `json:"is_public"`
	UserID            string `json:"user_id"`
	FinetuneGPUCount  int64  `json:"finetune_gpu_count"`
	InferenceGPUCount int64  `json:"inference_gpu_count"`
}

type ClusterCpodsResp struct {
	Data []CpodInfo `json:"data"`
}

type ClusterNode struct {
	NodeRole    string `json:"node_role"`
	NodeName    string `json:"node_name"`
	NodeIP      string `json:"node_ip"`
	SSHPort     int    `json:"ssh_port"`
	SSHUser     string `json:"ssh_user"`
	SSHPassword string `json:"ssh_password"`
}

type ClusterNodeInfo struct {
	GpuCount   int      `json:"gpu_count"`
	GpuProduct string   `json:"gpu_product"`
	Name       string   `json:"name"`
	Role       []string `json:"role"`
}

type CpodInfo struct {
	CpodID         string `json:"cpod_id"`
	CpodVersion    string `json:"cpod_version"`
	GPUVendor      string `json:"gpu_vendor"`
	GPUProd        string `json:"gpu_prod"`
	GPUMem         int64  `json:"gpu_mem"`
	GPUTotal       int64  `json:"gpu_total"`
	GPUAllocatable int64  `json:"gpu_allocatable"`
	CreateTime     string `json:"create_time"`
	UpdateTime     string `json:"update_time"`
}

type CpodJobReq struct {
	CpodId string `form:"cpodid"`
}

type CpodJobResp struct {
	JobList              []map[string]interface{} `json:"job_list"`
	InferenceServiceList []InferenceService       `json:"inference_service_list"`
	JupyterlabList       []JupyterLab             `json:"jupyter_lab_list"`
}

type CreateNewUserIDResp struct {
	Message string `json:"message"`
}

type DiskInfo struct {
	Size  int `json:"size"`
	Usage int `json:"usage"`
}

type FinetuneReq struct {
	TrainingFile     string                 `json:"training_file"`
	DatasetIsPublic  bool                   `json:"dataset_is_public"`
	Model            string                 `json:"model"`
	ModelIsPublic    bool                   `json:"model_is_public"`
	GpuModel         string                 `json:"gpu_model,optional"`
	GpuCount         int64                  `json:"gpu_count,optional"`
	TrainedModelName string                 `json:"trainedModelName,optional"`
	Hyperparameters  map[string]interface{} `json:"hyperparameters,optional,omitempty"`
	Config           map[string]interface{} `json:"config,optional,omitempty"`
	UserID           string                 `header:"Sx-User-ID"`
}

type FinetuneResp struct {
	JobId string `json:"job_id"`
}

type GPUInfo struct {
	CUDA    string `json:"cuda"`
	Prod    string `json:"prod"`
	Driver  string `json:"driver"`
	Vendor  string `json:"vendor"`
	MemSize int    `json:"mem_size"`
	Status  string `json:"status"`
}

type GPUStateInfo struct {
	Temp      int    `json:"temp"`
	MemUsage  int    `json:"mem_usage"`
	GPUUsage  int    `json:"gpu_usage"`
	Power     int    `json:"power"`
	Status    string `json:"status"`
	Allocated bool   `json:"allocated"`
}

type GPUSummary struct {
	Allocatable int    `json:"allocatable"`
	Total       int    `json:"total"`
	Prod        string `json:"prod"`
	MemSize     int    `json:"mem_size"`
	Vendor      string `json:"vendor"`
}

type GPUTypeReq struct {
}

type GPUTypeResp struct {
	Amount         float64 `json:"amount"`
	GPUProd        string  `json:"gpuProd"`
	GPUAllocatable int64   `json:"gpuAllocatable"`
}

type ImageDelReq struct {
	UserID    string `json:"user_id"`
	ImageName string `json:"image_name"`
	Tag       string `json:"tag"`
}

type InferenceDeleteReq struct {
	ServiceName string `form:"service_name"`
	UserID      string `header:"Sx-User-ID"`
}

type InferenceDeleteResp struct {
	Message string `json:"message"`
}

type InferenceDeployReq struct {
	ModelName string `json:"model_name"`
	GpuModel  string `json:"gpu_model,optional"`
	GpuCount  int64  `json:"gpu_count,optional"`
	UserID    string `header:"Sx-User-ID"`
}

type InferenceDeployResp struct {
	ServiceName string `json:"service_name"`
}

type InferenceInfoReq struct {
	ServiceName string `json:"service_name,optional"`
	UserID      string `header:"Sx-User-ID"`
}

type InferenceInfoResp struct {
	Data []SysInference `json:"data"`
}

type InferenceService struct {
	ServiceName   string `json:"service_name"`
	Status        string `json:"status"`
	ModelName     string `json:"model_name"`
	ModelId       string `json:"model_id"`
	ModelSize     int64  `json:"model_size"`
	ModelIsPublic bool   `json:"model_is_public"`
	GpuType       string `json:"gpu_type"`
	GpuNumber     int64  `json:"gpu_number"`
	Template      string `json:"template"`
	CpodId        string `json:"cpod_id"`
	UserId        string `json:"user_id"`
}

type InferenceStatus struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
	URL         string `json:"url"`
}

type Job struct {
	CkptPath            string `json:"ckptPath"`
	CkptVol             string `json:"ckptVol"`
	CpodId              string `json:"cpodId"`
	CreateTime          string `json:"createTime"`
	DatasetName         string `json:"datasetName,omitempty"`
	DatasetPath         string `json:"datasetPath,omitempty"`
	Deleted             int    `json:"deleted"`
	GpuNumber           int    `json:"gpuNumber"`
	GpuType             string `json:"gpuType"`
	ImagePath           string `json:"imagePath"`
	JobId               int    `json:"jobId"`
	JobName             string `json:"jobName"`
	JobType             string `json:"jobType"`
	JsonAll             string `json:"jsonAll"`
	ModelPath           string `json:"modelPath"`
	ModelVol            string `json:"modelVol"`
	ObtainStatus        int    `json:"obtainStatus"`
	PretrainedModelName string `json:"pretrainedModelName,omitempty"`
	PretrainedModelPath string `json:"pretrainedModelPath,omitempty"`
	RunCommand          string `json:"runCommand,omitempty"`
	StopTime            int    `json:"stopTime"`
	StopType            int    `json:"stopType"`
	UpdateTime          string `json:"updateTime"`
	UserId              string `json:"userId"`
	WorkStatus          int    `json:"workStatus"`
}

type JobCallBackReq struct {
	Status string `json:"status"`
	URL    string `json:"url"`
	JobID  string `json:"jobId"`
}

type JobCreateReq struct {
	GpuNumber               int64             `json:"gpuNumber"`
	GpuType                 string            `json:"gpuType"`
	CkptPath                string            `json:"ckptPath"`
	CkptVol                 int64             `json:"ckptVol"`
	ModelPath               string            `json:"modelPath"`
	ModelVol                int64             `json:"modelVol"`
	ImagePath               string            `json:"imagePath"`
	JobType                 string            `json:"jobType"`
	StopType                int64             `json:"stopType,optional"`
	StopTime                int64             `json:"stopTime,optional"`
	PretrainedModelId       string            `json:"pretrainedModelId,optional"`
	PretrainedModelName     string            `json:"pretrainedModelName,optional"`
	PretrainedModelPath     string            `json:"pretrainedModelPath,optional"`
	PretrainedModelIsPublic bool              `json:"modelIsPublic,optional"`
	DatasetId               string            `json:"datasetId,optional"`
	DatasetName             string            `json:"datasetName,optional"`
	DatasetPath             string            `json:"datasetPath,optional"`
	DatasetIsPublic         bool              `json:"datasetIsPublic,optional"`
	TrainedModelName        string            `json:"trainedModelName,optional"`
	RunCommand              string            `json:"runCommand,optional"`
	CallbackUrl             string            `json:"callbackUrl,optional,omitempty"`
	Env                     map[string]string `json:"env,optional,omitempty"`
	UserID                  string            `header:"Sx-User-ID"`
}

type JobCreateResp struct {
	JobId string `json:"job_id"`
}

type JobGetReq struct {
	Current int    `form:"current"`
	Size    int    `form:"size"`
	UserID  string `header:"Sx-User-ID"`
}

type JobGetResp struct {
	Content       []Job `json:"content"`
	TotalElements int64 `json:"totalElements"`
}

type JobStatus struct {
	JobStatus string `json:"job_status"`
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	JobType   string `json:"jobtype"`
}

type JobStatusReq struct {
	JobId string `json:"job_id"`
}

type JobStatusResp struct {
	URL    string `json:"url"`
	Status string `json:"status"`
}

type JobStopReq struct {
	JobId string `json:"job_id"`
}

type JobStopResp struct {
	Message string `json:"message"`
}

type JobsDelReq struct {
	UserID string `header:"Sx-User-ID"`
	ToUser string `json:"user_id"`
}

type JobsDelResp struct {
	Message string `json:"message"`
}

type JupyterLab struct {
	JobName          string             `json:"jobName"`
	InstanceName     string             `json:"instanceName"`
	CPUCount         string             `json:"cpuCount"`
	Memory           string             `json:"memory"`
	GPUCount         int                `json:"gpuCount"`
	GPUProduct       string             `json:"gpuProduct"`
	DataVolumeSize   string             `json:"dataVolumeSize"`
	PretrainedModels []PretrainedModels `json:"pretrainedModels"`
	UserID           string             `json:"userId"`
}

type Jupyterlab struct {
	ID             int64  `json:"id,optional"`
	JobName        string `json:"job_name,optional"`
	InstanceName   string `json:"instance_name"`
	CPUCount       int64  `json:"cpu_count"`
	Memory         int64  `json:"memory"`
	GPUCount       int64  `json:"gpu_count,optional"`
	GPUProduct     string `json:"gpu_product,optional"`
	DataVolumeSize int64  `json:"data_volume_size"`
	ModelId        string `json:"model_id,optional"`
	ModelName      string `json:"model_name,optional"`
	ModelPath      string `json:"model_path,optional"`
	URL            string `json:"url,optional"`
	UserId         string `json:"user_id"`
	Status         string `json:"status,optional"`
}

type JupyterlabCreateReq struct {
	Jupyterlab
	UserID string `header:"Sx-User-ID"`
}

type JupyterlabCreateResp struct {
	Message string `json:"message"`
}

type JupyterlabDeleteReq struct {
	JobName string `json:"job_name"`
	UserID  string `header:"Sx-User-ID"`
}

type JupyterlabDeleteResp struct {
	Message string `json:"message"`
}

type JupyterlabImage struct {
	CreatedAt string `json:"created_at"`
	FullName  string `json:"full_name"`
	ImageName string `json:"image_name"`
	UpdatedAt string `json:"updated_at"`
}

type JupyterlabImageCreateReq struct {
	UserID       string `header:"Sx-User-ID"`
	BaseImage    string `json:"base_image"`
	InstanceName string `json:"instance_name"`
	JobName      string `json:"job_name"`
}

type JupyterlabImageCreateResp struct {
	Message string `json:"message"`
}

type JupyterlabImageDelReq struct {
	UserID    string `header:"Sx-User-ID"`
	ImageName string `json:"image_name"`
	TagName   string `json:"tag_name,optional"`
}

type JupyterlabImageDelResp struct {
	Message string `json:"message"`
}

type JupyterlabImageListReq struct {
	UserID string `header:"Sx-User-ID"`
}

type JupyterlabImageListResp struct {
	Data []JupyterlabImage `json:"data"`
}

type JupyterlabImageVersion struct {
	ImageName string `json:"image_name"`
	ImageSize int    `json:"image_size"`
	TagName   string `json:"tag_name"`
	FullName  string `json:"full_name"`
	PushTime  string `json:"push_time"`
}

type JupyterlabImageVersionListReq struct {
	UserID       string `header:"Sx-User-ID"`
	InstanceName string `form:"instance_name"`
}

type JupyterlabImageVersionListResp struct {
	Data []JupyterlabImageVersion `json:"data"`
}

type JupyterlabListReq struct {
	UserID string `header:"Sx-User-ID"`
}

type JupyterlabListResp struct {
	Data []Jupyterlab `json:"data"`
}

type JupyterlabStatus struct {
	JobName string `json:"job_name"`
	Status  string `json:"status"`
	URL     string `json:"url"`
}

type LoginReq struct {
	Password string `json:"password"`
	Username string `json:"username"`
}

type LoginResp struct {
	User  WrapUser `json:"user"` // TODO 去掉多余嵌套
	Token string   `json:"token"`
}

type MemInfo struct {
	Size  int `json:"size"`
	Usage int `json:"usage"`
}

type ModelUrlReq struct {
	DownloadUrls []string `json:"download_urls"`
	JobName      string   `json:"job_name"`
}

type ModelUrlResp struct {
	Message string `json:"message"`
}

type NetworkInfo struct {
	Throughput int    `json:"throughput"`
	Type       string `json:"type"`
}

type NodeAddReq struct {
	ClusterNode
}

type NodeAddResp struct {
	Message string `json:"message"`
}

type NodeInfo struct {
	CPUInfo        CPUInfo        `json:"cpu_info"`
	LinuxDist      string         `json:"linux_dist"`
	GPUInfo        GPUInfo        `json:"gpu_info"`
	GPUTotal       int            `json:"gpu_total"`
	GPUAllocatable int            `json:"gpu_allocatable"`
	NetworkInfo    NetworkInfo    `json:"network_info"`
	GPUState       []GPUStateInfo `json:"gpu_state"`
	KernelVersion  string         `json:"kernel_version"`
	DiskInfo       DiskInfo       `json:"disk_info"`
	Name           string         `json:"name"`
	MemInfo        MemInfo        `json:"mem_info"`
	Arch           string         `json:"arch"`
	Status         string         `json:"status"`
}

type NodeListReq struct {
}

type NodeListResp struct {
	Data []ClusterNodeInfo `json:"data"`
}

type PretrainedModels struct {
	PretrainedModelId   string `json:"pretrainedModelId"`
	PretrainedModelName string `json:"pretrainedModelName"`
	PretrainedModelPath string `json:"pretrainedModelPath"`
}

type Quota struct {
	UserId   string `json:"user_id"`
	Resource string `json:"resource"`
	Quota    int64  `json:"quota"`
}

type QuotaAddReq struct {
	Quota
	UserID string `header:"Sx-User-ID"`
}

type QuotaAddResp struct {
	Message string `json:"message"`
}

type QuotaDeleteReq struct {
	Id     int64  `json:"id"`
	UserID string `header:"Sx-User-ID"`
}

type QuotaDeleteResp struct {
	Message string `json:"message"`
}

type QuotaListReq struct {
	UserID string `header:"Sx-User-ID"`
	ToUser string `json:"user_id,optional"`
}

type QuotaListResp struct {
	Data []QuotaResp `json:"data"`
}

type QuotaResp struct {
	Id int64 `json:"id"`
	Quota
}

type QuotaUpdateReq struct {
	Id     int64  `json:"id"`
	Quota  int64  `json:"quota"`
	UserID string `header:"Sx-User-ID"`
}

type QuotaUpdateResp struct {
	Message string `json:"message"`
}

type RegisterUserReq struct {
	Code         string `path:"code"`
	Username     string `json:"username"`
	Email        string `json:"email"`
	Enabled      int    `json:"enabled"`
	Password     string `json:"password"`
	UserType     int    `json:"userType,optional"`
	CompanyName  string `json:"companyName,optional"`
	CompanyPhone string `json:"companyPhone,optional"`
}

type Resource struct {
	ID                string   `json:"id"`
	Name              string   `json:"name"`
	Object            string   `json:"object"`
	Owner             string   `json:"owner"`
	Size              int64    `json:"size"`
	IsPublic          bool     `json:"is_public"`
	UserID            string   `json:"user_id"`
	Tag               []string `json:"tag"`
	Template          string   `json:"template"`
	FinetuneGPUCount  int      `json:"finetune_gpu_count"`
	InferenceGPUCount int      `json:"inference_gpu_count"`
}

type ResourceAdaptersReq struct {
	UserID string `header:"Sx-User-ID"`
}

type ResourceDatasetsReq struct {
	UserID string `header:"Sx-User-ID"`
}

type ResourceInfo struct {
	GPUSummaries []GPUSummary `json:"gpu_summaries"`
	CPODVersion  string       `json:"cpod_version"`
	Nodes        []NodeInfo   `json:"nodes"`
	CPODID       string       `json:"cpod_id"`
	Caches       []Cache      `json:"caches"`
}

type ResourceListResp struct {
	PublicList []Resource `json:"public_list"`
	UserList   []Resource `json:"user_list"`
}

type ResourceModelsReq struct {
	UserID string `header:"Sx-User-ID"`
}

type SendEmailReq struct {
	Email string `form:"email"`
}

type SysInference struct {
	ServiceName   string `json:"service_name"`
	Status        string `json:"status"`
	ModelName     string `json:"model_name"`
	ModelId       string `json:"model_id"`
	ModelSize     int64  `json:"model_size"`
	ModelIsPublic bool   `json:"model_is_public"`
	Url           string `json:"url"`
	StartTime     string `json:"start_time"` // 推理服务启动时间
	EndTime       string `json:"end_time"`   // 推理服务终止时间
}

type UploaderAccessReq struct {
	UserID string `header:"Sx-User-ID"`
}

type UploaderAccessResp struct {
	AccessID  string `json:"access_id"`
	AccessKey string `json:"access_key"`
	UserID    string `json:"user_id"`
}

type User struct {
	UserID   string `json:"user_id"`
	UserName string `json:"user_name"`
}

type UserBilling struct {
	BillingId     string  `json:"billing_id"`     // 账单ID
	UserId        string  `json:"user_id"`        // 用户ID
	Amount        float64 `json:"amount"`         // 消费金额
	BillingStatus int64   `json:"billing_status"` // 账单状态（0 未支付、1 已支付、2 欠费） TODO 返回字符串
	JobId         string  `json:"job_id"`         // 关联任务id
	JobType       string  `json:"job_type"`       // 关联任务类型（例如：finetune、inference）
	BillingTime   string  `json:"billing_time"`   // 账单生成时间
	PaymentTime   string  `json:"payment_time"`   // 支付时间
	Description   string  `json:"description"`    // 账单描述（可选，详细说明此次费用的具体内容）
}

type UserInfo struct {
	CompanyID  string `json:"companyId"`
	CreateBy   string `json:"createBy"`
	CreateTime string `json:"createTime"`
	Email      string `json:"email"`
	Enabled    bool   `json:"enabled"`
	ID         int64  `json:"id"`
	UserID     string `json:"user_id"`
	IsAdmin    bool   `json:"isAdmin"`
	UpdateBy   string `json:"updateBy"`
	UpdateTime string `json:"updateTime"`
	UserType   int    `json:"userType"`
	Username   string `json:"username"`
}

type UserInfoReq struct {
	UserID string `header:"Sx-User-ID"`
}

type UserInfoResp struct {
	User UserInfo `json:"user"`
}

type UserListResp struct {
	Data []User `json:"data"`
}

type WrapUser struct {
	User UserInfo `json:"user"`
}
