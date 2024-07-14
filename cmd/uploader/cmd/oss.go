package cmd

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
)

var (
	resource  string
	dir       string
	userID    string
	endpoint  string
	accessID  string
	accessKey string
	bucket    string
)

// ossCmd represents the oss command, which interacts with SXWL.AI Cloud's object storage system.
var ossCmd = &cobra.Command{
	Use:   "oss",
	Short: "upload resource to oss",
	Long:  "upload resource to oss",
	Run: func(cmd *cobra.Command, args []string) {
		if !fs.IsDirExist(dir) {
			fmt.Println("Please input a correct local dir")
			os.Exit(1)
		}

		if userID == "" {
			fmt.Println("Please input a correct userID")
			os.Exit(1)
		}

		if accessID == "" || accessKey == "" {
			fmt.Println("Please input correct oss accessID and accessKey")
			os.Exit(1)
		}

		switch resource {
		case consts.Model, consts.Dataset, consts.Adapter:
		default:
			fmt.Println("data_type should be [model|dataset|adapter]")
			os.Exit(1)
		}

		storage.InitClient(accessID, accessKey)

		start := time.Now()
		prefix := ""
		switch resource {
		case consts.Model:
			prefix = fmt.Sprintf(consts.OSSUserModelPath, userID)
		case consts.Dataset:
			prefix = fmt.Sprintf(consts.OSSUserDatasetPath, userID)
		case consts.Adapter:
			prefix = fmt.Sprintf(consts.OSSUserAdapterPath, userID)
		}

		size, err := storage.UploadDir(bucket, dir, path.Join(prefix, filepath.Base(dir)), false)
		if err != nil {
			fmt.Printf("upload failed, error: %v\n", err)
			os.Exit(1)
		} else {
			fmt.Printf("%s has been uploaded. size: %s used time: %s\n", dir, fs.FormatBytes(size), time.Since(start).String())
		}
	},
}

func init() {
	rootCmd.AddCommand(ossCmd)
	ossCmd.Flags().SortFlags = false
	ossCmd.Flags().StringVarP(&resource, "resource", "r", "model", "[model|dataset|adapter]")
	ossCmd.Flags().StringVarP(&dir, "dir", "d", "/data", "local dir to be uploaded")
	ossCmd.Flags().StringVarP(&userID, "userid", "u", "", "userID")
	ossCmd.Flags().StringVar(&endpoint, "endpoint", "https://oss-cn-beijing.aliyuncs.com", "oss endpoint")
	ossCmd.Flags().StringVar(&accessID, "access_id", "", "oss accessId")
	ossCmd.Flags().StringVar(&accessKey, "access_key", "", "oss accessKey")
	ossCmd.Flags().StringVarP(&bucket, "bucket", "b", "sxwl-cache", "oss bucket")
}
