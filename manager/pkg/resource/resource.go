package resource

import (
	"strconv"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"

	"log"
)

// 单个GPU的信息状态信息
type GPUState struct {
	Status    string `json:"status"`    //GPU运转状态： normal\abnormal
	Allocated bool   `json:"allocated"` //是否已经被K8S调度器分配使用
	MemUsage  int    `json:"mem_usage"` //显存使用率 MB
	GPUUsage  int    `json:"gpu_usage"` //算力使用率 % （0——100）
	Power     int    `json:"power"`     //功率 W
	Temp      int    `json:"temp"`      //温度 摄氏度
}

// 一个节点（主机）的GPU相关信息（GPU可能有多个）
type GPUInfo struct {
	Status  string `json:"status"`   //GPU总体运转状态：normal(有可用的GPU) \ abnormal(GPU全部掉线)
	Vendor  string `json:"vendor"`   //制造商：  如 NVidia
	Prod    string `json:"prod"`     //产品型号： 如 H100
	Driver  string `json:"driver"`   //驱动版本
	CUDA    string `json:"cuda"`     //CUDA版本
	MemSize int    `json:"mem_size"` //MB
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
	Name           string                `json:"name"`           //节点名 hostname
	Status         string                `json:"status"`         //K8S系统中的节点状态
	Arch           string                `json:"arch"`           //指令架构
	KernelVersion  string                `json:"kernel_version"` //Linux内核版本
	LinuxDist      string                `json:"linux_dist"`     //Linux发行版信息  如：CentOS7.6
	GPUInfo        `json:"gpu_info"`     //GPU信息
	GPUAllocatable int                   `json:"gpu_allocatable"` //可用GPU总量（暂时先不管个体信息）
	GPUState       []GPUState            `json:"gpu_state"`       //GPU状态信息对应到每张卡
	CPUInfo        `json:"cpu_info"`     //CPU信息
	MemInfo        `json:"mem_info"`     //内存信息
	DiskInfo       `json:"disk_info"`    //磁盘信息
	NetworkInfo    `json:"network_info"` //网络信息
}

// 对于CPod的资源描述：包含资源总量，使用情况，分配情况
type CPodResourceInfo struct {
	CPodID      string     `json:"cpod_id"`      //CPOD_ID , 对于CPod的唯一标识
	CPodVersion string     `json:"cpod_version"` //CPod的版本， 这里会隐含K8S的版本信息
	Nodes       []NodeInfo `json:"nodes"`        //节点信息
}

func GetResourceInfo(CPodID string, CPodVersion string) CPodResourceInfo {
	var info CPodResourceInfo
	info.CPodID = CPodID
	info.CPodVersion = CPodVersion
	// get node list from k8s
	info.Nodes = []NodeInfo{}
	if nodeInfo, err := clientgo.GetNodeInfo(); err == nil {
		for _, node := range nodeInfo.Items {
			t := NodeInfo{}
			t.Name = node.Name
			t.Status = node.Labels["status"]
			t.KernelVersion = node.Labels["feature.node.kubernetes.io/kernel-version.full"]
			t.LinuxDist = node.Status.NodeInfo.OSImage
			t.Arch = node.Labels["kubernetes.io/arch"]
			t.CPUInfo.Cores = int(node.Status.Capacity.Cpu().Value())
			if v, ok := node.Labels["nvidia.com/gpu.product"]; ok {
				t.GPUInfo.Prod = v
				t.GPUInfo.Vendor = "nvidia"
				t.GPUInfo.Driver = node.Labels["nvidia.com/cuda.driver.major"] + "." +
					node.Labels["nvidia.com/cuda.driver.minor"] + "." +
					node.Labels["nvidia.com/cuda.driver.rev"]
				t.GPUInfo.CUDA = node.Labels["nvidia.com/cuda.runtime.major"] + "." +
					node.Labels["nvidia.com/cuda.runtime.minor"]
				t.GPUInfo.MemSize, _ = strconv.Atoi(node.Labels["nvidia.com/gpu.memory"])
				t.GPUInfo.Status = "abnormal"
				if node.Labels["nvidia.com/gpu.present"] == "true" {
					t.GPUInfo.Status = "normal"
				}
				//init GPUState Array , accordding to nvidia.com/gpu.count label
				t.GPUState = []GPUState{}
				gpuCnt, _ := strconv.Atoi(node.Labels["nvidia.com/gpu.count"])
				for i := 0; i < gpuCnt; i++ {
					t.GPUState = append(t.GPUState, GPUState{})
				}
				tmp := node.Status.Allocatable["nvidia.com/gpu"].DeepCopy()
				if i, ok := (&tmp).AsInt64(); ok {
					t.GPUAllocatable = int(i)
				}
			}
			t.MemInfo.Size = int(node.Status.Capacity.Memory().Value() / 1024 / 1024)
			info.Nodes = append(info.Nodes, t)
		}
	} else {
		log.Println("failed retrieve node info from k8s.")
	}
	// TODO: 从监控系统获取信息并补充到info，从APIServer获取的信息中没有网络以及磁盘的信息，需要从监控系统中获取
	return info
}
