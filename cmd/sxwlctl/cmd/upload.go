package cmd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sxwl/3k/cmd/sxwlctl/internal/oss"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
)

var (
	dir     string
	typ     string
	authURL string
)

// uploadCmd represents the upload command
var uploadCmd = &cobra.Command{
	Use:   "upload",
	Short: "upload",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		if !fs.IsDirExist(dir) {
			fmt.Println("Please input a correct local dir")
			os.Exit(1)
		}

		switch typ {
		case "model", "dataset":
		default:
			fmt.Println("data_type should be [model|dataset]")
			os.Exit(1)
		}

		if token == "" {
			fmt.Println("Please input a correct sxwl token")
			os.Exit(1)
		}

		accessID, accessKey, userID, err := getAccessByToken(token)
		if err != nil {
			fmt.Println("Please input a correct sxwl token")
			os.Exit(1)
		}

		conf := oss.Config{
			Endpoint:  "https://oss-cn-beijing.aliyuncs.com",
			AccessID:  accessID,
			AccessKey: accessKey,
			Bucket:    "sxwl-cache",
		}

		// init oss client
		storage.InitClient(accessID, accessKey)

		start := time.Now()
		prefix := ""
		switch typ {
		case "model":
			prefix = fmt.Sprintf("models/user-%d/", userID)
		case "dataset":
			prefix = fmt.Sprintf("datasets/user-%d/", userID)
		}
		size, err := upload(conf, path.Join(prefix, filepath.Base(dir)), dir)
		if err != nil {
			fmt.Printf("upload err: %s\n", err)
			os.Exit(1)
		} else {
			fmt.Printf("%s has been uploaded. size: %s used time: %s\n", dir, fs.FormatBytes(size), time.Since(start).String())
		}
	},
}

func init() {
	rootCmd.AddCommand(uploadCmd)
	uploadCmd.Flags().StringVarP(&dir, "dir", "d", "", "上传的本地文件夹")
	uploadCmd.Flags().StringVar(&typ, "data_type", "model", "[model|dataset]")
	uploadCmd.Flags().StringVar(&authURL, "auth_url", "https://llm.sxwl.ai/api/uploader_access", "鉴权接口")
}

func getAccessByToken(token string) (id, key string, userID int64, err error) {
	// Create a new request using http.NewRequest
	req, err := http.NewRequest("GET", authURL, nil)
	if err != nil {
		err = fmt.Errorf("Error creating request: %s\n", err)
		return
	}

	// Add an 'Authorization' header to the request
	req.Header.Add("Authorization", "Bearer "+token)

	// Send the request using http.Client
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		err = fmt.Errorf("Error sending request to API endpoint: %s\n", err)
		return
	}
	defer resp.Body.Close()

	// Check if the response status code indicates success (200 OK)
	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("auth_url request failed with status code: %d", resp.StatusCode)
		return
	}

	// Decode the JSON response into the UploaderAccessResp struct
	var response types.UploaderAccessResp
	if err = json.NewDecoder(resp.Body).Decode(&response); err != nil {
		err = fmt.Errorf("Error decoding JSON response: %s\n", err)
		return
	}

	return response.AccessID, response.AccessKey, response.UserID, err
}

func upload(conf oss.Config, prefix, localDir string) (int64, error) {
	return storage.UploadDirToOSS(conf.Bucket, prefix, localDir)
}
