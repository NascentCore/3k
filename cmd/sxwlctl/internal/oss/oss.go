package oss

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/aliyun/aliyun-oss-go-sdk/oss"
)

type Config struct {
	Endpoint  string
	AccessID  string
	AccessKey string
	Bucket    string
}

type Client struct {
	conf Config
}

func NewClient(c Config) *Client {
	return &Client{c}
}

func (c *Client) UploadLocal(dir string, dest string) error {
	// Initialize OSS client
	client, err := oss.New(c.conf.Endpoint, c.conf.AccessID, c.conf.AccessKey)
	if err != nil {
		return fmt.Errorf("failed to create OSS client: %v", err)
	}

	// Get the specified bucket
	bucket, err := client.Bucket(c.conf.Bucket)
	if err != nil {
		return fmt.Errorf("failed to get bucket %s: %v", c.conf.Bucket, err)
	}

	// Walk through the directory and upload each file
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("error accessing path %q: %v", path, err)
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Generate the OSS object key based on the file relative path and the specified destination
		relativePath, err := filepath.Rel(dir, path)
		if err != nil {
			return fmt.Errorf("failed to calculate relative path for %s: %v", path, err)
		}
		objectKey := filepath.Join(dest, relativePath)

		// Upload the file
		err = bucket.PutObjectFromFile(objectKey, path)
		if err != nil {
			return fmt.Errorf("failed to upload %s to %s: %v", path, objectKey, err)
		}
		fmt.Printf("Successfully uploaded %s to %s\n", path, objectKey)
		return nil
	})

	return err
}

// func (conf *Client) Download() error {
//     // 初始化oss
//     conf := conf.conf.OSSConfig
//     client, err := oss.New(conf.Endpoint, conf.AccessID, conf.AccessKey)
//     if err != nil {
//         return fmt.Errorf("error initializing OSS client err: %s", err)
//     }
//
//     // Get bucket
//     bucket, err := client.Bucket(conf.Bucket)
//     if err != nil {
//         return fmt.Errorf("error getting bucket: %s err: %s", conf.Bucket, err)
//     }
//
//     // List objects with the given prefix
//     lsRes, err := bucket.ListObjects(oss.Prefix(conf.Object))
//     if err != nil {
//         return fmt.Errorf("error listing objects err: %s", err)
//     }
//
//     isFile := false
//     prefix := conf.Object
//     if len(lsRes.Objects) == 1 {
//         isFile = true
//     }
//
//     for _, object := range lsRes.Objects {
//         // Skip directories (objects ending with "/")
//         if object.Key[len(object.Key)-1] == '/' {
//             continue
//         }
//
//         // Construct the relative path by removing the prefix from the object key
//         relativePath := strings.TrimPrefix(object.Key, prefix)
//
//         // Construct local file path using the relative path
//         localFilePath := filepath.Join(conf.conf.OutDir, relativePath)
//         if isFile {
//             // 下载单个文件
//             _, filename := filepath.Split(object.Key)
//             localFilePath = filepath.Join(conf.conf.OutDir, filename)
//         }
//
//         // Ensure local directory structure exists
//         if err := os.MkdirAll(filepath.Dir(localFilePath), os.ModePerm); err != nil {
//             return fmt.Errorf("error creating local directory for object: %s err: %s", object.Key, err)
//         }
//
//         // Download the object
//         err := bucket.GetObjectToFile(object.Key, localFilePath)
//         if err != nil {
//             return fmt.Errorf("error downloading object: %s err: %s", object.Key, err)
//         }
//     }
//
//     return nil
// }
