package storage

import (
	"bufio"
	"fmt"
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
func UploadFileToOSS(bucketName, userID, jobName, modelPath, statePath string) error {
	logfilePath := path.Join(statePath, config.FILE_UPLOAD_LOG_FILE)
	signedUrlPath := path.Join(statePath, config.PRESIGNED_URL_FILE)
	objectKey := path.Join("models", userID, jobName)
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		return err
	}

	err = filepath.Walk(path.Join(modelPath), func(subPath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".md") {
			return nil
		}
		relativePath, err := filepath.Rel(path.Join(modelPath), subPath)
		if err != nil {
			return err
		}

		// 指定上传选项，启用断点续传功能
		err = bucket.UploadFile(filepath.Join(objectKey, relativePath), subPath, 100*1024, oss.Routines(3), oss.Checkpoint(true, ""))
		if err != nil {
			return fmt.Errorf("failed to upload file %v", err)
		}

		return nil
	})
	if err != nil {
		return err
	}

	// 获取 presigned url
	signedURL, err := bucket.SignURL(objectKey, oss.HTTPGet, int64(config.OSS_URL_EXPIRED_SECOND))
	if err != nil {
		return err
	}
	err = WriteFile(signedUrlPath, signedURL)
	if err != nil {
		return err
	}
	err = WriteFile(logfilePath, jobName)
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
		err := UploadFileToOSS(bucket, prefix, file, dir, "")
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
	listOptions := []oss.Option{
		oss.Prefix(prefix),
		oss.MaxKeys(1000),
	}
	result, err := bucket.ListObjects(listOptions...)
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

// ListFiles return the files and size match the prefix
func ListFiles(bucketName, prefix string) (map[string]int64, error) {
	// Get the OSS bucket
	bucket, err := client.Bucket(bucketName)
	if err != nil {
		log.SLogger.Errorw("error getting bucket", "bucketName", bucketName, "error", err)
		return nil, err
	}

	// Initialize the map to return
	fileSizes := make(map[string]int64)

	// Define list options for the OSS API call
	listOptions := []oss.Option{
		oss.Prefix(prefix), // List files with the specified prefix
	}

	// Use a loop to handle pagination in case there are more than 1000 objects
	isTruncated := true
	for isTruncated {
		result, err := bucket.ListObjects(listOptions...)
		if err != nil {
			log.SLogger.Errorw("error listing objects", "prefix", prefix, "error", err)
			return nil, err
		}

		// Process each object in the list results
		for _, object := range result.Objects {
			fileSizes[object.Key] = object.Size
		}

		isTruncated = result.IsTruncated
		// If there are more objects, set the next marker for the next list call
		if isTruncated {
			listOptions = append(listOptions, oss.Marker(result.NextMarker))
		}
	}

	return fileSizes, nil
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

// ExistFile checks if oss://bucketName/filePath exists. Int64 is the file size in bytes.
func ExistFile(bucketName, filePath string) (bool, int64, error) {
	// Create an OSS client for the specified bucket
	svc, err := client.Bucket(bucketName)
	if err != nil {
		return false, 0, err
	}

	// List objects with the exact file path as prefix.
	// Since file paths are unique, if the file exists, it should be the only one listed.
	result, err := svc.ListObjects(oss.Prefix(filePath), oss.MaxKeys(1))
	if err != nil {
		return false, 0, err
	}

	// Check if the file exists by looking at the objects returned
	if len(result.Objects) == 1 && result.Objects[0].Key == filePath {
		// If the object exists and the key matches the file path, return true with the file size
		return true, result.Objects[0].Size, nil
	}

	// If the object was not found or the key does not match exactly, the file does not exist
	return false, 0, nil
}

// UploadDir uploads the whole local directory to the OSS
func UploadDir(bucketName, localDirPath, ossDirPath string, verbose bool) (int64, error) {
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

		if verbose {
			totalBytes += info.Size()
			log.SLogger.Infow("file uploaded", "ossPath", ossPath, "filePath", filePath, "size", info.Size())
		}

		return nil
	})

	if err != nil {
		log.SLogger.Errorw("walk local dir err", "error", err)
		return totalBytes, err
	}

	return totalBytes, nil
}

// DownloadDir will download the ossDirPath to localDirPath and returns the disk size downloaded and the error.
func DownloadDir(bucketName, localDirPath, ossDirPath string, verbose bool) (int64, error) {
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
	// List objects in the OSS directory
	listOptions := []oss.Option{
		oss.Prefix(ossDirPath),
	}
	isTruncated := true

	for isTruncated {
		// Get the list of objects
		result, err := bucket.ListObjects(listOptions...)
		if err != nil {
			log.SLogger.Errorw("list objects err", "error", err)
			return totalBytes, err
		}

		for _, object := range result.Objects {
			// Calculate the local file path
			relativePath := strings.TrimPrefix(object.Key, ossDirPath)
			localFilePath := filepath.Join(localDirPath, relativePath)

			// Ensure the local directory exists
			if err := os.MkdirAll(filepath.Dir(localFilePath), os.ModePerm); err != nil {
				log.SLogger.Errorw("create local dir err", "error", err, "localFilePath", localFilePath)
				return totalBytes, err
			}

			// Download the file
			err = bucket.GetObjectToFile(object.Key, localFilePath)
			if err != nil {
				log.SLogger.Errorw("download file err", "error", err, "ossPath", object.Key, "localFilePath", localFilePath)
				return totalBytes, err
			}

			// Always calculate the total bytes downloaded
			fileInfo, err := os.Stat(localFilePath)
			if err != nil {
				log.SLogger.Errorw("stat file err", "error", err, "localFilePath", localFilePath)
				return totalBytes, err
			}
			totalBytes += fileInfo.Size()

			if verbose {
				log.SLogger.Infow("file downloaded", "ossPath", object.Key, "localFilePath", localFilePath, "size", fileInfo.Size())
			}
		}

		isTruncated = result.IsTruncated
		if isTruncated {
			listOptions = append(listOptions, oss.Marker(result.NextMarker))
		}
	}

	return totalBytes, nil
}
