package main

import (
	"fmt"
	"os"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	modeluploader "sxwl/3k/manager/pkg/model-uploader"
)

// NO_TEST_NEEDED

// Input ： MPIJob的Name ， S3的地址以及Bucket
func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage : modeluploadjob  [ MPIJob Name ] [ Bucket ] ")
		os.Exit(1)
	}
	mpiJobName := os.Args[1]
	bucket := os.Args[2]
	//检查上传任务是否已经开始
	if started, err := modeluploader.CheckUploadStarted(modeluploader.UploadStartedFlagFile); err != nil {
		os.Exit(1)
	} else if !started { //尚末开始
		clientgo.InitClient()
		//等待MPI Job完成并取得其最终的状态
		status, err := modeluploader.UntilMPIJobFinish(mpiJobName)
		//如果发生错误，进程异常退出
		if err != nil {
			os.Exit(1)
		}
		//训练出现问题，没有模型可以上传，正常结束
		if status != modeluploader.StatusNeedUploadModel {
			return
		}
		//写入开始上传标志
		if err := modeluploader.MarkUploadStarted(modeluploader.UploadStartedFlagFile); err != nil {
			os.Exit(1)
		}
	}
	//（继续）上传模型
	//如果发生错误，进程异常退出
	if err := modeluploader.UploadModel(bucket); err != nil {
		os.Exit(1)
	}
}
