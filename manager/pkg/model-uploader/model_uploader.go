package modeluploader

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"sxwl/3k/manager/pkg/config"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/storage"
	"time"
)

// moniter the given mpijob , until it stops (fail , finish , mannual interrupt or whatever)
// 返回 (KUBEFLOW)MPIJob结束时的状态 status  和  监控过程中发生的程序必须中断不可恢复的错误 err
// 当status为“”时，err必不为nil , status不为“”时，err必为nil。
// 当status为“Complete”时，代表需要进一步执行模型上传工作。
// 当err != nil 时，代表监控无法继续进行，需要由K8S触发重启。
func UntilMPIJobFinish(mpiJobName string) (string, error) {
	errCnt := 0
	for {
		s, err := kubeflowmpijob.GetState(config.CPOD_NAMESPACE, mpiJobName)
		if err != nil {
			//如果MPIJob已经被删除
			if errors.Is(err, kubeflowmpijob.ErrNotFound) {
				log.SLogger.Infow("mpijob is deleted")
				//if job is deleted , do upload as well
				return config.STATUS_NEEDS_UPLOAD_MODEL, nil
				//return "deleted", nil
			}
			log.SLogger.Errorw("error when wait mpijob finish", "error", err, "errcnt", errCnt)
			errCnt += 1
			//持续报错， 返回异常
			if errCnt > 10 {
				return "", errors.New("cant get state in 10 minutes")
			}
		} else { //errCnt置0
			errCnt = 0
		}
		log.SLogger.Infow("mpijob status", "jobname", mpiJobName, "status", s.JobStatus)
		//判断MPIJob是否已经终止（状态不会再变）
		if s.JobStatus.NoMoreChange() {
			if s.JobStatus == state.JobStatusSucceed {
				log.SLogger.Infow("mpijob succeed")
				return config.STATUS_NEEDS_UPLOAD_MODEL, nil
			}
			log.SLogger.Infow("mpijob failed , nothing to upload")
			return "nouploadneeded", nil
		}
		//继续等待
		log.SLogger.Infow("keep waiting")
		time.Sleep(time.Minute)
	}
}

// err != nil 代表上传任务失败，程序无法继续执行。
// 需要由K8S触发重启
func UploadModel(bucket string, jobName string, modelPath string) error {
	return storage.UploadDirToOSS(bucket, jobName, modelPath)
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

func PostUrlsToMarket(fileName, jobName, url string) error {
	lines, err := GetUrls(fileName)
	if err != nil {
		return err
	}

	data := map[string]interface{}{
		"job_name":      jobName,
		"download_urls": lines,
	}
	body, err := json.Marshal(data)
	if err != nil {
		return err
	}

	r, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		return err
	}

	r.Header.Add("Content-Type", "application/json")
	r.Header.Add("Authorization", "Bearer "+config.ACCESS_KEY_MARKET)

	client := &http.Client{}
	res, err := client.Do(r)
	if err != nil {
		return err
	}
	if res.StatusCode != 200 {
		return errors.New("post presigned url to market failed")
	}

	return nil
}

func GetUrls(fileName string) ([]string, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
	}

	if err = scanner.Err(); err != nil {
		return nil, err
	}

	return lines, nil
}
