package config

import "os"

var (
	STORAGE_CLASS_TO_CREATE_PVC = os.Getenv("StorageClass") //创建PVC所指定的SC，在CPod集群中需要提前创建好
	MODELUPLOADER_IMAGE         = os.Getenv("UPLOADIMAGE")  //ModelUploader的Docker Image ， 需要在Cpod集群上能够Pull
	OSS_ACCESS_KEY              = os.Getenv("AK")
	OSS_ACCESS_SECRET           = os.Getenv("AS")
	DEPLOY                      = os.Getenv("DEPLOY")
)
