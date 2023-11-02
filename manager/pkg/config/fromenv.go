package config

import "os"

var (
	//for cpod manager
	STORAGE_CLASS_TO_CREATE_PVC = os.Getenv("StorageClass") //创建PVC所指定的SC，在CPod集群中需要提前创建好
	MODELUPLOADER_IMAGE         = os.Getenv("UPLOADIMAGE")  //ModelUploader的Docker Image ， 需要在Cpod集群上能够Pull
	DEPLOY                      = os.Getenv("DEPLOY")
	//for model uploader
	OSS_ACCESS_KEY    = os.Getenv(OSS_ACCESS_KEY_ENV_NAME)
	OSS_ACCESS_SECRET = os.Getenv(OSS_ACCESS_SECRET_ENV_NAME)
	OSS_BUCKET        = func() string { //OSS Bucket
		if DEPLOY == "PROD" {
			return "sxwl-ai"
		}
		return "sxwl-ai-test"
	}()
)
