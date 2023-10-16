package main

import (
	"fmt"
	"os"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
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
	clientgo.InitClient()

	//等待MPI Job完成并取得其最终的状态
	status := UntilMPIJobFinish(mpiJobName)
	if status != "Completed" {
		//训练出现问题，没有模型可以上传
		return
	}
	//上传模型
	err := UploadModel(bucket)
	//如果发生错误，进程异常退出，让K8 S去重启Pod，重新尝试
	if err != nil {
		os.Exit(1)
	}

}
