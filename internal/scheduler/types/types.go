// Code generated by goctl. DO NOT EDIT.
package types

type AuthReq struct {
	UserID int64 `header:"Sx-User"`
}

type CPODStatusReq struct {
	JobStatus       []JobStatus       `json:"job_status"`
	InferenceStatus []InferenceStatus `json:"inference_status"`
	ResourceInfo    ResourceInfo      `json:"resource_info"`
	UpdateTime      string            `json:"update_time"`
	CPODID          string            `json:"cpod_id"`
	UserID          int64             `header:"Sx-User"`
}

type CPODStatusResp struct {
	Message string `json:"message"`
}

type CPUInfo struct {
	Cores int `json:"cores"`
	Usage int `json:"usage"`
}

type Cache struct {
	DataType   string `json:"data_type"`
	DataName   string `json:"data_name"`
	DataId     string `json:"data_id"`
	DataSize   int64  `json:"data_size"`
	Template   string `json:"template"`
	DataSource string `json:"data_source"`
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

type CpodJobReq struct {
	CpodId string `form:"cpodid"`
}

type CpodJobResp struct {
	JobList              []map[string]interface{} `json:"job_list"`
	InferenceServiceList []InferenceService       `json:"inference_service_list"`
}

type DiskInfo struct {
	Size  int `json:"size"`
	Usage int `json:"usage"`
}

type FinetuneReq struct {
	TrainingFile     string                 `json:"training_file"`
	Model            string                 `json:"model"`
	GpuModel         string                 `json:"gpu_model,optional"`
	GpuCount         int64                  `json:"gpu_count,optional"`
	TrainedModelName string                 `json:"trainedModelName,optional"`
	Hyperparameters  map[string]interface{} `json:"hyperparameters,optional,omitempty"`
	Config           map[string]interface{} `json:"config,optional,omitempty"`
	UserID           int64                  `header:"Sx-User" json:"-"`
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

type InferenceDeleteReq struct {
	ServiceName string `form:"service_name"`
	UserID      int64  `header:"Sx-User"`
}

type InferenceDeleteResp struct {
	Message string `json:"message"`
}

type InferenceDeployReq struct {
	ModelName string `json:"model_name"`
	GpuModel  string `json:"gpu_model,optional"`
	GpuCount  int64  `json:"gpu_count,optional"`
	UserID    int64  `header:"Sx-User"`
}

type InferenceDeployResp struct {
	ServiceName string `json:"service_name"`
}

type InferenceInfoReq struct {
	ServiceName string `json:"service_name,optional"`
	UserID      int64  `header:"Sx-User"`
}

type InferenceInfoResp struct {
	Data []SysInference `json:"data"`
}

type InferenceService struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
	ModelName   string `json:"model_name"`
	ModelId     string `json:"model_id"`
	ModelSize   int64  `json:"model_size"`
	GpuType     string `json:"gpu_type"`
	GpuNumber   int64  `json:"gpu_number"`
	Template    string `json:"template"`
	CpodId      string `json:"cpod_id"`
	UserId      int64  `json:"user_id"`
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
	UserId              int    `json:"userId"`
	WorkStatus          int    `json:"workStatus"`
}

type JobCallBackReq struct {
	Status string `json:"status"`
	URL    string `json:"url"`
	JobID  string `json:"jobId"`
}

type JobCreateReq struct {
	GpuNumber           int64             `json:"gpuNumber"`
	GpuType             string            `json:"gpuType"`
	CkptPath            string            `json:"ckptPath"`
	CkptVol             int64             `json:"ckptVol"`
	ModelPath           string            `json:"modelPath"`
	ModelVol            int64             `json:"modelVol"`
	ImagePath           string            `json:"imagePath"`
	JobType             string            `json:"jobType"`
	StopType            int64             `json:"stopType,optional"`
	StopTime            int64             `json:"stopTime,optional"`
	PretrainedModelId   string            `json:"pretrainedModelId,optional"`
	PretrainedModelName string            `json:"pretrainedModelName,optional"`
	PretrainedModelPath string            `json:"pretrainedModelPath,optional"`
	DatasetId           string            `json:"datasetId,optional"`
	DatasetName         string            `json:"datasetName,optional"`
	DatasetPath         string            `json:"datasetPath,optional"`
	TrainedModelName    string            `json:"trainedModelName,optional"`
	RunCommand          string            `json:"runCommand,optional"`
	CallbackUrl         string            `json:"callbackUrl,optional,omitempty"`
	Env                 map[string]string `json:"env,optional,omitempty"`
	UserID              int64             `header:"Sx-User" json:"-"`
}

type JobCreateResp struct {
	JobId string `json:"job_id"`
}

type JobGetReq struct {
	Current int   `form:"current"`
	Size    int   `form:"size"`
	UserID  int64 `header:"Sx-User"`
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

type LoginReq struct {
	Password string `json:"password"`
	Username string `json:"username"`
}

type LoginResp struct {
	User  WrapUser `json:user` // TODO 去掉多余嵌套
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

type Quota struct {
	UserId   int64  `json:"user_id"`
	Resource string `json:"resource"`
	Quota    int64  `json:"quota"`
}

type QuotaAddReq struct {
	Quota
	UserID int64 `header:"Sx-User"`
}

type QuotaAddResp struct {
	Message string `json:"message"`
}

type QuotaDeleteReq struct {
	Id     int64 `json:"id"`
	UserID int64 `header:"Sx-User"`
}

type QuotaDeleteResp struct {
	Message string `json:"message"`
}

type QuotaListReq struct {
	UserId int64 `json:"user_id,optional"`
}

type QuotaListResp struct {
	Data []QuotaResp `json:"data"`
}

type QuotaResp struct {
	Id int64 `json:"id"`
	Quota
}

type QuotaUpdateReq struct {
	Id     int64 `json:"id"`
	Quota  int64 `json:"quota"`
	UserID int64 `header:"Sx-User"`
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
	ID     string   `json:"id"`
	Name   string   `json:"name"`
	Object string   `json:"object"`
	Owner  string   `json:"owner"`
	Size   int64    `json:"size"`
	Tag    []string `json:"tag"`
}

type ResourceDatasetsReq struct {
	UserID int64 `header:"Sx-User" json:"-"`
}

type ResourceDatasetsResp struct {
	Total int64      `json:"total"`
	List  []Resource `json:"list"`
}

type ResourceInfo struct {
	GPUSummaries []GPUSummary `json:"gpu_summaries"`
	CPODVersion  string       `json:"cpod_version"`
	Nodes        []NodeInfo   `json:"nodes"`
	CPODID       string       `json:"cpod_id"`
	Caches       []Cache      `json:"caches"`
}

type ResourceModelsReq struct {
	UserID int64 `header:"Sx-User" json:"-"`
}

type ResourceModelsResp struct {
	Total int64      `json:"total"`
	List  []Resource `json:"list"`
}

type SendEmailReq struct {
	Email string `form:"email"`
}

type SysInference struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
	ModelName   string `json:"model_name"`
	ModelId     string `json:"model_id"`
	ModelSize   int64  `json:"model_size"`
	Url         string `json:"url"`
	StartTime   string `json:"start_time"` // 推理服务启动时间
	EndTime     string `json:"end_time"`   // 推理服务终止时间
}

type UploaderAccessReq struct {
	UserID int64 `header:"Sx-User" json:"-"`
}

type UploaderAccessResp struct {
	AccessID  string `json:"access_id"`
	AccessKey string `json:"access_key"`
	UserID    int64  `json:"user_id"`
}

type User struct {
	UserId   int64  `json:"user_id"`
	UserName string `json:"user_name"`
}

type UserInfo struct {
	CompanyID  string `json:"companyId"`
	CreateBy   string `json:"createBy"`
	CreateTime string `json:"createTime"`
	Email      string `json:"email"`
	Enabled    bool   `json:"enabled"`
	ID         int    `json:"id"`
	IsAdmin    bool   `json:"isAdmin"`
	Password   string `json:"password"`
	UpdateBy   string `json:"updateBy"`
	UpdateTime string `json:"updateTime"`
	UserType   int    `json:"userType"`
	Username   string `json:"username"`
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
