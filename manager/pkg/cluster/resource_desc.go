package cluster

type GPUDesc struct {
	//TODO
}

type CPUDesc struct {
	//TODO
}

type MemDesc struct {
	//TODO
}

type StorageDesc struct {
	//TODO
}

type NetworkDesc struct {
	//TODO
}

type NodeDesc struct {
	GPUDesc
	CPUDesc
	MemDesc
	StorageDesc
	NetworkDesc
}

type ClusterDesc struct {
	//TODO
}

type ResourceDesc struct {
	Nodes []NodeDesc
	ClusterDesc
}
