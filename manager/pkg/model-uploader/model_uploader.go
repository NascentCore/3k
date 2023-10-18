package modeluploader

// NO_TEST_NEEDED

// moniter the given mpijob , until it stops (fail , finish , mannual interrupt or whatever)
// 返回 MPIJob结束时的状态 status  和  监控过程中发生的程序必须中断不可恢复的错误 err
// 当status为“”时，err必不为nil , status不为“”时，err必为nil。
// 当status为“Complete”时，代表需要进一步执行模型上传工作。
// 当err != nil 时，代表监控无法继续进行，需要由K8S触发重启。
func UntilMPIJobFinish(mpiJobName string) (string, error) {
	return "Complete", nil
}

// err != nil 代表上传任务失败，程序无法继续执行。
// 需要由K8S触发重启
func UploadModel(bucket string) error {
	return nil
}
