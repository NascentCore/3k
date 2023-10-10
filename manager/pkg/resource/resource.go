package resource

// 单个GPU的信息状态信息
type GPUStatus struct {
	Status   string `json:"status"`    //GPU运转状态： normal\abnormal
	MemUsage int    `json:"mem_usage"` //MB
	GPUUsage int    `json:"gpu_usage"` //%
}

// 一个节点（主机）的GPU相关信息（GPU可能有多个）
type GPUInfo struct {
	Status      string      `json:"status"`      //GPU总体运转状态：normal(有可用的GPU) \ abnormal(GPU全部掉线)
	Vendor      string      `json:"vendor"`      //制造商：  如 NVidia
	Prod        string      `json:"prod"`        //产品型号： 如 H100
	Driver      string      `json:"driver"`      //驱动版本
	MemSize     int         `json:"mem_size"`    //MB
	Allocated   int         `json:"allocated"`   //已经分配在用的数量 0——len(individuals)
	Individuals []GPUStatus `json:"individuals"` //按个体的状态统计
}

// 节点的CPU相关信息
type CPUInfo struct {
	Cores int `json:"cores"` //逻辑核心数量
	Usage int `json:"usage"` //使用率（%） 0——100
}

// 节点的内存相关信息
type MemInfo struct {
	Size  int `json:"size"`  //内存大小 MB
	Usage int `json:"usage"` //内存使用 MB
}

// 节点的磁盘相关信息
type DiskInfo struct {
	Size  int `json:"size"`  //容器所在磁盘大小 MB
	Usage int `json:"usage"` //使用量 MB
}

// 节点的网络相关信息
type NetworkInfo struct {
	Type       string `json:"type"`       // 网卡类型 如IB
	Throughput int    `json:"throughput"` // 网络吞吐量 MB/s
}

// 节点信息
type NodeInfo struct {
	Status        string                `json:"status"`         //节点状态： normal\abnormal
	KernelVersion string                `json:"kernel_version"` //Linux内核版本
	LinuxDist     string                `json:"linux_dist"`     //Linux发行版信息  如：CentOS7.6
	GPUInfo       `json:"gpu_info"`     //GPU信息
	CPUInfo       `json:"cpu_info"`     //CPU信息
	MemInfo       `json:"mem_info"`     //内存信息
	DiskInfo      `json:"disk_info"`    //磁盘信息
	NetworkInfo   `json:"network_info"` //网络信息
}

// 对于CPod的资源描述：包含资源总量，使用情况，分配情况
type CPodResourceInfo struct {
	CPodID      string     `json:"cpod_id"`      //CPOD_ID , 对于CPod的唯一标识
	CPodVersion string     `json:"cpod_version"` //CPod的版本， 这里会隐含K8S的版本信息
	Nodes       []NodeInfo `json:"nodes"`        //节点信息
}

func GetResourceInfo() CPodResourceInfo {
	// TODO: get resource info from cluster and monitor system
	return CPodResourceInfo{}
}
