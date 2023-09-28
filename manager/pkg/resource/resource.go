package resource

type GPUDesc struct {
	// TODO: fill the struct
}

type CPUDesc struct {
	// TODO: fill the struct
}

type MemDesc struct {
	// TODO: fill the struct
}

type StorageDesc struct {
	// TODO: fill the struct
}

type NetworkDesc struct {
	// TODO: fill the struct
}

type NodeDesc struct {
	GPUDesc
	CPUDesc
	MemDesc
	StorageDesc
	NetworkDesc
}

type ClusterDesc struct {
	// TODO: fill the struct
}

type ResourceDesc struct {
	Nodes []NodeDesc
	ClusterDesc
}

func GetResourceDesc() ResourceDesc {
	// TODO: get resource info from cluster and monitor system
	return ResourceDesc{}
}
