package cmd

import (
	"fmt"
	"os"
	"sxwl/3k/cmd/sxwlctl/internal/sxy"
	"sxwl/3k/pkg/consts"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	source string
	resID  string
	resType string
	statusCmd bool
)

// loadCmd represents the load command
var loadCmd = &cobra.Command{
	Use:   "load",
	Short: "load resource from external source",
	Long:  `从外部源(如huggingface)加载资源到系统中`,
	Run: func(cmd *cobra.Command, args []string) {
		// 检查token
		token := viper.GetString("token")
		if token == "" {
			fmt.Println("Please input a sxwl token")
			os.Exit(1)
		}

		if statusCmd {
			// 查询资源加载状态
			resp, err := sxy.GetResourceSyncTaskStatus(token)
			if err != nil {
				fmt.Printf("get resource sync task status failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("\n找到 %d 个资源同步任务:\n", resp.Total)
			fmt.Println("----------------------------------------")
			for _, task := range resp.Data {
				fmt.Printf("资源ID:   \t%s\n", task.ResourceID)
				fmt.Printf("资源类型: \t%s\n", task.ResourceType) 
				fmt.Printf("来源:     \t%s\n", task.Source)
				fmt.Printf("状态:     \t%s\n", task.Status)
				fmt.Println("----------------------------------------")
			}
			return
		}

		// 检查必要参数
		if source == "" {
			fmt.Println("Please specify source with -s")
			os.Exit(1)
		}
		if resID == "" {
			fmt.Println("Please specify resource id with -i") 
			os.Exit(1)
		}

		// 检查资源类型
		switch resType {
		case consts.Model, consts.Dataset, consts.Adapter:
		default:
			fmt.Println("resource_type should be [model|dataset|adapter]")
			os.Exit(1)
		}

		// 调用resource/load接口
		err := sxy.LoadResource(token, sxy.LoadResourceReq{
			Source:       source,
			ResourceID:   resID,
			ResourceType: resType,
		})
		if err != nil {
			fmt.Printf("load resource failed: resource_id=%s, type=%s, source=%s, error=%v\n", resID, resType, source, err)
			os.Exit(1)
		}

		fmt.Printf("Resource load task created successfully. Source: %s, ID: %s, Type: %s\n", 
			source, resID, resType)
	},
}

func init() {
	rootCmd.AddCommand(loadCmd)
	loadCmd.Flags().SortFlags = false
	loadCmd.Flags().StringVarP(&source, "source", "s", "", "资源来源[huggingface|modelscope]")
	loadCmd.Flags().StringVarP(&resType, "type", "t", "model", "资源类型[model|dataset]")
	loadCmd.Flags().StringVarP(&resID, "id", "i", "", "资源ID 例如: meta-llama/Meta-Llama-3.1-8B")
	loadCmd.Flags().BoolVar(&statusCmd, "status", false, "查询资源加载状态")
}
