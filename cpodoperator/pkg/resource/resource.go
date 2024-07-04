package resource

// NO_TEST_NEEDED
const (
	CacheModel   = "model"
	CacheDataSet = "dataset"
	CacheImage   = "image"
	CacheAdapter = "adapter"
)

// 单个GPU的信息状态信息
type GPUState struct {
	Status    string `json:"status"`    // GPU运转状态： normal\abnormal
	Allocated bool   `json:"allocated"` // 是否已经被K8S调度器分配使用
	MemUsage  int    `json:"mem_usage"` // 显存使用率 MB
	GPUUsage  int    `json:"gpu_usage"` // 算力使用率 % （0——100）
	Power     int    `json:"power"`     // 功率 W
	Temp      int    `json:"temp"`      // 温度 摄氏度
}

// 一个节点（主机）的GPU相关信息（GPU可能有多个）
type GPUInfo struct {
	Status  string `json:"status"`   // GPU总体运转状态：normal(有可用的GPU) \ abnormal(GPU全部掉线)
	Vendor  string `json:"vendor"`   // 制造商：  如 NVidia
	Prod    string `json:"prod"`     // 产品型号： 如 H100
	Driver  string `json:"driver"`   // 驱动版本
	CUDA    string `json:"cuda"`     // CUDA版本
	MemSize int    `json:"mem_size"` // MB
}

// 节点的CPU相关信息
type CPUInfo struct {
	Cores int `json:"cores"` // 逻辑核心数量
	Used  int `json:"used"`  // 使用核心数量
	Usage int `json:"usage"` // 使用率（%） 0——100
}

// 节点的内存相关信息
type MemInfo struct {
	Size  int `json:"size"`  // 内存大小 MB
	Used  int `json:"used"`  // 内存使用 MB
	Usage int `json:"usage"` // 内存使用 MB
}

// 节点的磁盘相关信息
type DiskInfo struct {
	Size  int `json:"size"`  // 容器所在磁盘大小 MB
	Usage int `json:"usage"` // 使用量 MB
}

// 节点的网络相关信息
type NetworkInfo struct {
	Type       string `json:"type"`       // 网卡类型 如IB
	Throughput int    `json:"throughput"` // 网络吞吐量 MB/s
}

// 节点信息
type NodeInfo struct {
	Name           string                `json:"name"`           // 节点名 hostname
	Status         string                `json:"status"`         // K8S系统中的节点状态
	Arch           string                `json:"arch"`           // 指令架构
	KernelVersion  string                `json:"kernel_version"` // Linux内核版本
	LinuxDist      string                `json:"linux_dist"`     // Linux发行版信息  如：CentOS7.6
	GPUInfo        `json:"gpu_info"`     // GPU信息
	GPUTotal       int                   `json:"gpu_total"`       // GPU总量
	GPUAllocatable int                   `json:"gpu_allocatable"` // 可用GPU总量（暂时先不管个体信息）
	GPUUsed        int                   `json:"gpu_used"`        // 已用GPU总量（暂时先不管个体信息）
	GPUState       []GPUState            `json:"gpu_state"`       // GPU状态信息对应到每张卡
	CPUInfo        `json:"cpu_info"`     // CPU信息
	MemInfo        `json:"mem_info"`     // 内存信息
	DiskInfo       `json:"disk_info"`    // 磁盘信息
	NetworkInfo    `json:"network_info"` // 网络信息
}

// 某一类GPU的汇总信息
type GPUSummary struct {
	Vendor      string `json:"vendor"`      // 生产商
	Prod        string `json:"prod"`        // 型号
	MemSize     int    `json:"mem_size"`    // MB
	Total       int    `json:"total"`       // 总量
	Allocatable int    `json:"allocatable"` // 可用量
}

// 对于CPod的资源描述：包含资源总量，使用情况，分配情况
type CPodResourceInfo struct {
	CPodID       string       `json:"cpod_id"`       // CPOD_ID , 对于CPod的唯一标识
	CPodVersion  string       `json:"cpod_version"`  // CPod的版本， 这里会隐含K8S的版本信息
	GPUSummaries []GPUSummary `json:"gpu_summaries"` // GPU汇总信息，每一条代表一类GPU（如3090）
	Caches       []Cache      `json:"caches"`
	Nodes        []NodeInfo   `json:"nodes"` // 节点信息
}

// “缓存”在三千平台上的各类数据：开源基底模型、数据集
// TODO：目前系统还没有考虑用户自有的模型、数据集，须注意此类设计上的安全性
type Cache struct {
	IsPublic          bool   `json:"is_public"`
	UserID            string `json:"user_id"`
	DataType          string `json:"data_type"`
	DataName          string `json:"data_name"`
	DataId            string `json:"data_id"`
	DataSize          int64  `json:"data_size"`
	Template          string `json:"template"`
	DataSource        string `json:"data_source"`
	FinetuneGPUCount  int64  `json:"finetune_gpu_count"`
	InferenceGPUCount int64  `json:"inference_gpu_count"`
}
