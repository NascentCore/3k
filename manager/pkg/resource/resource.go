package resource

import "encoding/json"

type GPU struct {
	Status   string `json:"status"`    //GPU运转状态： normal\abnormal
	Driver   string `json:"driver"`    //驱动版本
	MemUsage int    `json:"mem_usage"` //MB
	GPUUsage int    `json:"gpu_usage"` //%
}

type GPUs struct {
	Status      string `json:"status"`      //GPU总体运转状态：normal(有可用的GPU) \ abnormal(GPU全部掉线)
	Vendor      string `json:"vendor"`      //制造商：  如 NVidia
	Prod        string `json:"prod"`        //产品型号： 如 H100
	MemSize     int    `json:"mem_size"`    //MB
	Allocated   int    `json:"allocated"`   //已经分配在用的数量 0——len(individuals)
	Individuals []GPU  `json:"individuals"` //按个体的状态统计
}

type CPU struct {
	Cores int `json:"cores"` //逻辑核心数量
	Usage int `json:"usage"` //使用率（%） 0——100
}

type Mem struct {
	Size  int `json:"size"`  //内存大小 MB
	Usage int `json:"usage"` //内存使用 MB
}

type Disk struct {
	Size  int `json:"size"`  //容器所在磁盘大小 MB
	Usage int `json:"usage"` //使用量 MB
}

type Network struct {
	Type       string `json:"type"`       // 网卡类型 如IB
	Throughput int    `json:"throughput"` // 网络吞吐量 MB/s
}

type Node struct {
	Status        string           `json:"status"`         //节点状态： normal\abnormal
	KernelVersion string           `json:"kernel_version"` //Linux内核版本
	LinuxDist     string           `json:"linux_dist"`     //Linux发行版信息  如：CentOS7.6
	GPUs          `json:"gpus"`    //GPU信息
	CPU           `json:"cpu"`     //CPU信息
	Mem           `json:"mem"`     //内存信息
	Disk          `json:"disk"`    //磁盘信息
	Network       `json:"network"` //网络信息
}

// 对于CPod的资源描述：包含资源总量，使用情况，分配情况
type CPodResourceDesc struct {
	CPodID      string `json:"cpod_id"`      //CPOD_ID , 对于CPod的唯一标识
	CPodVersion string `json:"cpod_version"` //CPod的版本， 这里会隐含K8S的版本信息
	Nodes       []Node `json:"nodes"`        //节点信息
}

func (rd CPodResourceDesc) ToJson() []byte {
	res, _ := json.Marshal(rd)
	return res
}

func GetResourceDesc() CPodResourceDesc {
	// TODO: get resource info from cluster and monitor system
	return CPodResourceDesc{}
}
