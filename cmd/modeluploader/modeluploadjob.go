package main

import (
	"fmt"
	"os"
	"path"

	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/log"
	modeluploader "sxwl/3k/pkg/model-uploader"
	"sxwl/3k/pkg/storage"
)

// NO_TEST_NEEDED

// Input ： MPIJob的Name ， S3的地址以及Bucket
func main() {
	// TODO: with cli tools
	if len(os.Args) != 4 {
		fmt.Println("Usage : modeluploadjob  [userID] [ Modelstorage ] [ Bucket ]")
		os.Exit(1)
	}
	userID := os.Args[1]
	modelstorage := os.Args[2]
	bucket := os.Args[3]
	//检查上传任务是否已经开始
	if started, err := modeluploader.CheckUploadStarted(path.Join(config.MODELUPLOADER_STATE_PATH, config.UPLOAD_STARTED_FLAG_FILE)); err != nil {
		log.SLogger.Infow("checkupload started failed", "error", err)
		os.Exit(1)
	} else if !started { //尚末开始
		//写入开始上传标志
		if err := modeluploader.MarkUploadStarted(path.Join(config.MODELUPLOADER_STATE_PATH, config.UPLOAD_STARTED_FLAG_FILE)); err != nil {
			log.SLogger.Infow("error when mark upload started", "error", err)
			os.Exit(1)
		}
	}
	//（继续）上传模型
	//add access key before build
	storage.InitClient(config.OSS_ACCESS_KEY, config.OSS_ACCESS_SECRET)
	if err := modeluploader.UploadPackedFile(bucket, userID, modelstorage); err != nil {
		log.SLogger.Infow("upload model error", "error", err)
		os.Exit(1)
	}
	if err := modeluploader.PostUrlsToMarket(path.Join(config.MODELUPLOADER_STATE_PATH, config.PRESIGNED_URL_FILE), modelstorage, config.BASE_URL+config.URLPATH_UPLOAD_URLS); err != nil {
		log.SLogger.Infow("post presigned urls to market error", "error", err)
		os.Exit(1)
	}
	log.SLogger.Infow("upload model done , job finish")
}
