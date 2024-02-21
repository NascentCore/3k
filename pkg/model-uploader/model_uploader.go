package modeluploader

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/log"
	"sxwl/3k/pkg/storage"
)

// err != nil 代表上传任务失败，程序无法继续执行。
// 需要由K8S触发重启
func UploadModel(bucket string, jobName string, modelPath string) error {
	_, err := storage.UploadDirToOSS(bucket, jobName, modelPath)
	return err
}

func UploadPackedFile(bucket string, jobName string) error {
	return storage.UploadFileToOSS(bucket, jobName, config.PACK_FILE_NAME, config.MODELUPLOADER_PVC_MOUNT_PATH)
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
	_, err := os.Stat(fileName)
	if err != nil && os.IsNotExist(err) {
		log.SLogger.Infow("not found ", fileName, ", post aborted")
		return nil
	}
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
		return fmt.Errorf("post presigned url to market failed , statuscode: %d", res.StatusCode)
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
