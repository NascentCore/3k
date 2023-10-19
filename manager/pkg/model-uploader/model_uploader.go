package modeluploader

import "os"

// NO_TEST_NEEDED

const StatusNeedUploadModel = "Complete"
const UploadStartedFlagFile = "/data/upload_started_flag_file"

// moniter the given mpijob , until it stops (fail , finish , mannual interrupt or whatever)
// 返回 MPIJob结束时的状态 status  和  监控过程中发生的程序必须中断不可恢复的错误 err
// 当status为“”时，err必不为nil , status不为“”时，err必为nil。
// 当status为“Complete”时，代表需要进一步执行模型上传工作。
// 当err != nil 时，代表监控无法继续进行，需要由K8S触发重启。
func UntilMPIJobFinish(mpiJobName string) (string, error) {
	return StatusNeedUploadModel, nil
}

// err != nil 代表上传任务失败，程序无法继续执行。
// 需要由K8S触发重启
func UploadModel(bucket string) error {
	//上传
	return nil
}

// 标记上传开始
func MarkUploadStarted(fileName string) error {
	return os.WriteFile(fileName, []byte("let's go"), os.ModePerm)
}

// 在启动时检查是否已经开始上传（应对上传中断的情况）
func CheckUploadStarted(fileName string) (bool, error) {
	_, err := os.ReadFile(fileName)
	if err == nil {
		return true, nil
	} else if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
