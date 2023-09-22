package cluster

type GPUDesc struct {
}

type CPUDesc struct {
}

type MemDesc struct {
}

type StorageDesc struct {
}

type NetworkDesc struct {
}

type NodeDesc struct {
	GPUDesc
	CPUDesc
	MemDesc
	StorageDesc
	NetworkDesc
}

type ClusterDesc struct {
}

type ResourceDesc struct {
	Nodes []NodeDesc
	ClusterDesc
}
