package storage

import (
	"bufio"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/log"

	"github.com/aliyun/aliyun-oss-go-sdk/oss"
)

var client oss.Client

func InitClient(accessID, accessKey string) {
	cli, err := oss.New(config.OSS_ENDPOINT, accessID, accessKey)
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
			// 去除标识文件
			if item.Name() == config.FILE_UPLOAD_LOG_FILE || item.Name() == config.UPLOAD_STARTED_FLAG_FILE ||
				item.Name() == config.PRESIGNED_URL_FILE || item.Name() == config.PACK_FILE_NAME {
			} else {
				res = append(res, path.Join(prefix, item.Name()))
			}
		}
	}
	return res, nil
}

// 从日志文件中读取已上传文件列表
func GetUploaded(dir string) (map[string]struct{}, error) {
	file, err := os.Open(path.Join(dir, config.FILE_UPLOAD_LOG_FILE))
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
	logfilePath := path.Join(dir, config.FILE_UPLOAD_LOG_FILE)
	signedUrlPath := path.Join(dir, config.PRESIGNED_URL_FILE)
	ossPath := path.Join(pathPrefix, filePath)
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		return err
	}
	err = bucket.UploadFile(ossPath, path.Join(dir, filePath), 100*1024, oss.Routines(3), oss.Checkpoint(true, ""))
	if err != nil {
		return err
	}
	// 获取 presigned url
	signedURL, err := bucket.SignURL(ossPath, oss.HTTPGet, int64(config.OSS_URL_EXPIRED_SECOND))
	if err != nil {
		return err
	}
	err = WriteFile(signedUrlPath, signedURL)
	if err != nil {
		return err
	}
	err = WriteFile(logfilePath, filePath)
	if err != nil {
		return err
	}
	return nil
}

func UploadDirToOSS(bucket, prefix, dir string) (int64, error) {
	// files are without dir
	files, err := FilesToUpload(dir, "")
	if err != nil {
		log.SLogger.Errorw("get file list error", "error", err)
		return 0, err
	}
	uploaded, err := GetUploaded(dir)
	if err != nil {
		log.SLogger.Errorw("get uploaded file list err", "error", err)
		return 0, err
	}

	size, err := fs.GetDirSize(dir)
	if err != nil {
		log.SLogger.Errorw("get dir size err", "error", err)
		return 0, err
	}

	for _, file := range files {
		// 已上传
		if _, ok := uploaded[file]; ok {
			log.SLogger.Infow("file already uploaded", "file", file)
			continue
		}
		// do upload
		log.SLogger.Infow("start uploading", "file", file)
		err := UploadFileToOSS(bucket, prefix, file, dir)
		if err != nil {
			log.SLogger.Errorw("upload to oss err", "error", err)
			return 0, err
		}
		log.SLogger.Infow("uploaded", "file", file)
	}
	return size, nil
}

func WriteFile(filePath, content string) error {
	f, err := os.OpenFile(filePath, os.O_WRONLY|os.O_APPEND|os.O_CREATE, os.ModePerm)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write([]byte(content + "\n"))
	if err != nil {
		return err
	}
	return nil
}

// ListDir list the dirs in the bucket with a prefix dir. And ignore all level is not sub.
func ListDir(bucketName, prefix string, sub int) (map[string]int64, error) {
	// Ensure the prefix ends with a slash to denote a directory-like structure
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	// Get the OSS bucket
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		log.SLogger.Errorw("error getting bucket", "bucketName", bucketName, "error", err)
		return nil, err
	}

	// ListDir objects with the specified prefix
	result, err := bucket.ListObjects(oss.Prefix(prefix))
	if err != nil {
		log.SLogger.Errorw("error listing objects", "prefix", prefix, "error", err)
		return nil, err
	}

	dirSizes := make(map[string]int64)
	for _, object := range result.Objects {
		// Calculate the directory path of the object
		dirPath := strings.TrimPrefix(object.Key, prefix)
		if strings.Count(dirPath, "/") != sub {
			continue
		}
		slashIndex := strings.LastIndex(dirPath, "/")
		if slashIndex > -1 {
			dirName := dirPath[:slashIndex+1]
			fullDirPath := prefix + dirName
			if _, exists := dirSizes[fullDirPath]; !exists {
				// Initialize size for new directories
				dirSizes[fullDirPath] = 0
			}
			// Sum the size of the object to its directory's total size
			dirSizes[fullDirPath] += object.Size
		}
	}

	// Since directories might not have explicit objects, handle them separately
	for _, commonPrefix := range result.CommonPrefixes {
		dirPath := strings.TrimPrefix(commonPrefix, prefix)
		fullDirPath := prefix + dirPath
		if _, exists := dirSizes[fullDirPath]; !exists {
			// Initialize size for directories without explicit objects
			dirSizes[fullDirPath] = 0
		}
	}

	return dirSizes, nil
}

// ExistDir checks if oss://bucketName/dirPath exists. Int64 is the dir size in bytes.
func ExistDir(bucketName, dirPath string) (bool, int64, error) {
	// Create an OSS client
	svc, err := client.Bucket(bucketName)
	if err != nil {
		return false, 0, err
	}

	// Add a trailing slash if dirPath does not end with one
	if !strings.HasSuffix(dirPath, "/") {
		dirPath += "/"
	}

	// List objects in the directory with the specified prefix
	result, err := svc.ListObjects(oss.Prefix(dirPath))
	if err != nil {
		return false, 0, err
	}

	// Initialize directory size
	var size int64 = 0

	// Iterate over objects to calculate total size
	for _, object := range result.Objects {
		size += object.Size
	}

	// If there are any objects, the directory exists
	return len(result.Objects) > 0, size, nil
}

// UploadDir uploads the whole local directory to the OSS
func UploadDir(bucketName, localDirPath, ossDirPath string) (int64, error) {
	// Ensure the OSS directory path ends with a slash
	if !strings.HasSuffix(ossDirPath, "/") {
		ossDirPath += "/"
	}

	// Get the OSS bucket
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		log.SLogger.Errorw("get bucket err", "error", err)
		return 0, err
	}

	var totalBytes int64
	// Walk through the local directory
	err = filepath.Walk(localDirPath, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Calculate the relative path and use it as the OSS object name
		relativePath, err := filepath.Rel(localDirPath, filePath)
		if err != nil {
			return err
		}
		ossPath := ossDirPath + relativePath

		// Upload the file to OSS
		err = bucket.PutObjectFromFile(ossPath, filePath)
		if err != nil {
			log.SLogger.Errorw("upload file to oss err", "error", err, "ossPath", ossPath, "filePath", filePath)
			return err
		}

		totalBytes += info.Size()
		log.SLogger.Infow("file uploaded", "ossPath", ossPath, "filePath", filePath, "size", info.Size())

		return nil
	})

	if err != nil {
		log.SLogger.Errorw("walk local dir err", "error", err)
		return totalBytes, err
	}

	return totalBytes, nil
}
