package storage

import (
	"bufio"
	"os"
	"path"
	"sxwl/3k/manager/pkg/log"

	"github.com/aliyun/aliyun-oss-go-sdk/oss"
)

const FilesUploadedLogFile = "files_uploaded_log"

var client oss.Client

func InitClient(accessID, accessKey string) {
	cli, err := oss.New("http://oss-cn-beijing.aliyuncs.com", accessID, accessKey)
	if err != nil {
		log.SLogger.Errorw("init oss client err", "error", err)
		os.Exit(1)
	}
	log.SLogger.Info("oss client inited")
	client = *cli
}

// list all files needs to upload , prefix 为当前的DIR所对应的前缀
func FilesToUpload(dir, prefix string) ([]string, error) {
	items, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	res := []string{}
	for _, item := range items {
		if item.IsDir() {
			files, err := FilesToUpload(path.Join(dir, item.Name()), path.Join(prefix, item.Name()))
			if err != nil {
				return nil, err
			}
			res = append(res, files...)
		} else {
			//去除标识文件
			if item.Name() == FilesUploadedLogFile || item.Name() == "upload_started_flag_file" {
			} else {
				res = append(res, path.Join(prefix, item.Name()))
			}
		}
	}
	return res, nil
}

// 从日志文件中读取已上传文件列表
func GetUploaded(dir string) (map[string]struct{}, error) {
	file, err := os.Open(path.Join(dir, FilesUploadedLogFile))
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]struct{}{}, nil
		} else {
			return nil, err
		}
	}
	defer file.Close()
	res := map[string]struct{}{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		res[scanner.Text()] = struct{}{}
	}
	if scanner.Err() != nil {
		return nil, err
	}
	return res, nil
}

// 上传文件到OSS，开启断点续传
func UploadFileToOSS(bucketName, pathPrefix, filePath, dir string) error {
	logfilePath := path.Join(dir, FilesUploadedLogFile)
	ossPath := path.Join(pathPrefix, filePath)
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		return err
	}
	err = bucket.UploadFile(ossPath, path.Join(dir, filePath), 100*1024, oss.Routines(3), oss.Checkpoint(true, ""))
	if err != nil {
		return err
	}
	f, err := os.OpenFile(logfilePath, os.O_WRONLY|os.O_APPEND|os.O_CREATE, os.ModePerm)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write([]byte(filePath + "\n"))
	if err != nil {
		return err
	}
	return nil
}

func UploadDirToOSS(bucket, prefix, dir string) error {
	//files are without dir
	files, err := FilesToUpload(dir, "")
	if err != nil {
		log.SLogger.Errorw("get file list error", "error", err)
		return err
	}
	uploaded, err := GetUploaded(dir)
	if err != nil {
		log.SLogger.Errorw("get uploaded file list err", "error", err)
		return err
	}
	for _, file := range files {
		//已上传
		if _, ok := uploaded[file]; ok {
			log.SLogger.Infow("file already uploaded", "file", file)
			continue
		}
		// do upload
		log.SLogger.Infow("start uploading", "file", file)
		err := UploadFileToOSS(bucket, prefix, file, dir)
		if err != nil {
			log.SLogger.Errorw("upload to oss err", "error", err)
			return err
		}
		log.SLogger.Infow("uploaded", "file", file)
	}
	return nil
}
