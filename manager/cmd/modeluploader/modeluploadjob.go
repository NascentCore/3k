package main

import (
	"fmt"
	"os"
	"path"

	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/config"
	"sxwl/3k/manager/pkg/log"
	modeluploader "sxwl/3k/manager/pkg/model-uploader"
	"sxwl/3k/manager/pkg/storage"
)

// NO_TEST_NEEDED

// Input ： MPIJob的Name ， S3的地址以及Bucket
func main() {
	// TODO: with cli tools
	if len(os.Args) != 3 {
		fmt.Println("Usage : modeluploadjob  [ MPIJob Name ] [ Bucket ]")
		os.Exit(1)
	}
	mpiJobName := os.Args[1]
	bucket := os.Args[2]
	//检查上传任务是否已经开始
	if started, err := modeluploader.CheckUploadStarted(path.Join(config.MODELUPLOADER_PVC_MOUNT_PATH, config.UPLOAD_STARTED_FLAG_FILE)); err != nil {
		log.SLogger.Infow("checkupload started failed", "error", err)
		os.Exit(1)
	} else if !started { //尚末开始
		clientgo.InitClient()
		//等待MPI Job完成并取得其最终的状态
		status, err := modeluploader.UntilMPIJobFinish(mpiJobName)
		//如果发生错误，进程异常退出
		if err != nil {
			log.SLogger.Infow("error when wait mpijob finish", "error", err)
			os.Exit(1)
		}
		//训练出现问题，没有模型可以上传，正常结束
		if status != config.STATUS_NEEDS_UPLOAD_MODEL {
			log.SLogger.Infow("nothing to upload , job finish")
			return
		}
		//写入开始上传标志
		if err := modeluploader.MarkUploadStarted(path.Join(config.MODELUPLOADER_PVC_MOUNT_PATH, config.UPLOAD_STARTED_FLAG_FILE)); err != nil {
			log.SLogger.Infow("error when mark upload started", "error", err)
			os.Exit(1)
		}
	}
	//（继续）上传模型
	//add access key before build
	storage.InitClient(config.OSS_ACCESS_KEY, config.OSS_ACCESS_SECRET)
	//如果发生错误，进程异常退出
	if err := modeluploader.UploadModel(bucket, mpiJobName, config.MODELUPLOADER_PVC_MOUNT_PATH); err != nil {
		log.SLogger.Infow("upload model error", "error", err)
		os.Exit(1)
	}
	log.SLogger.Infow("upload model done , job finish")
}
