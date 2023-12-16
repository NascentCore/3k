package config

// NO_TEST_NEEDED

import "os"

var (
	// for cpod manager
	// ModelUploader的Docker Image ， 需要在Cpod集群上能够Pull
	MODELUPLOADER_IMAGE = os.Getenv("UPLOADIMAGE")
	DEPLOY              = os.Getenv("DEPLOY")
	
	//for model uploader
	OSS_ACCESS_KEY    = os.Getenv(OSS_ACCESS_KEY_ENV_NAME)
	OSS_ACCESS_SECRET = os.Getenv(OSS_ACCESS_SECRET_ENV_NAME)
	
	// TODO: Should get BASE_URL from command line flags or config file, not derived from env var.
	OSS_BUCKET        = func() string { //OSS Bucket
		if DEPLOY == "PROD" {
			return "sxwl-ai"
		}
		return "sxwl-ai-test"
	}()
	
	// TODO: Should get BASE_URL from command line flags or config file, not derived from env var.
	STORAGE_CLASS_TO_CREATE_PVC = func() string { //创建PVC所指定的SC，在CPod集群中需要提前创建好
		if DEPLOY == "PROD" {
			return "cpod-cephfs"
		}
		return "ceph-filesystem"
	}()
	
	// TODO: Should get BASE_URL from command line flags or config file, not derived from env var.
	BASE_URL = func() string { //portal base url
		if DEPLOY == "PROD" {
			return "https://cloud.nascentcore.ai"
		}
		return "https://aiapi.yangapi.cn"
	}()
	
	ACCESS_KEY        = os.Getenv("ACCESS_KEY") //from configmap provided by cairong
	CPOD_ID           = os.Getenv("CPOD_ID")    //from configmap provided by cairong
	ACCESS_KEY_MARKET = os.Getenv(MARKET_ACCESS_KEY)
)
