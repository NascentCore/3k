package oss

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sxwl/3k/cmd/downloader/internal/download/config"

	"github.com/aliyun/aliyun-oss-go-sdk/oss"
)

type Downloader struct {
	c config.Config
}

func NewDownloader(c config.Config) *Downloader {
	return &Downloader{c}
}

func (d *Downloader) Download() error {
	// 初始化oss
	conf := d.c.OSSConfig
	client, err := oss.New(conf.Endpoint, conf.AccessID, conf.AccessKey)
	if err != nil {
		return fmt.Errorf("error initializing OSS client err: %s", err)
	}

	// Get bucket
	bucket, err := client.Bucket(conf.Bucket)
	if err != nil {
		return fmt.Errorf("error getting bucket: %s err: %s", conf.Bucket, err)
	}

	// List objects with the given prefix
	lsRes, err := bucket.ListObjects(oss.Prefix(conf.Object))
	if err != nil {
		return fmt.Errorf("error listing objects err: %s", err)
	}

	isFile := false
	prefix := conf.Object
	if len(lsRes.Objects) == 1 {
		isFile = true
	}

	for _, object := range lsRes.Objects {
		// Skip directories (objects ending with "/")
		if object.Key[len(object.Key)-1] == '/' {
			continue
		}

		// Construct the relative path by removing the prefix from the object key
		relativePath := strings.TrimPrefix(object.Key, prefix)

		// Construct local file path using the relative path
		localFilePath := filepath.Join(d.c.OutDir, relativePath)
		if isFile {
			// 下载单个文件
			_, filename := filepath.Split(object.Key)
			localFilePath = filepath.Join(d.c.OutDir, filename)
		}

		// Ensure local directory structure exists
		if err := os.MkdirAll(filepath.Dir(localFilePath), os.ModePerm); err != nil {
			return fmt.Errorf("error creating local directory for object: %s err: %s", object.Key, err)
		}

		// Download the object
		err := bucket.GetObjectToFile(object.Key, localFilePath)
		if err != nil {
			return fmt.Errorf("error downloading object: %s err: %s", object.Key, err)
		}
	}

	return nil
}
