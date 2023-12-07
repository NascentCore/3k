/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"fmt"
	"github.com/spf13/cobra"
	"os"
	"os/exec"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/fs"
)

var (
	gitUrl    string
	outDir    string
	group     string // "cpod.sxwl.ai"
	version   string // "v1"
	plural    string // "modelstorages"
	name      string // "example-modelstorage"
	namespace string // "cpod"
)

// gitCmd represents the git command
var gitCmd = &cobra.Command{
	Use:   "git",
	Short: "download data from git url",
	Long:  `download data from git url`,
	Run: func(cmd *cobra.Command, args []string) {
		clientgo.InitClient()

		if gitUrl == "" {
			fmt.Println("please input the git url for downloading")
			return
		}

		// empty dir /data
		if fs.IsDirExist(outDir) {
			err := fs.RemoveAllFilesInDir(outDir)
			if err != nil {
				fmt.Printf("Error RemoveAllFilesInDir %s err: %v\n", outDir, err)
				os.Exit(1)
			}
		}

		// download repo
		err := downloadGitRepo(gitUrl, outDir)
		if err != nil {
			fmt.Printf("Error downloading Git repository err: %v\n", err)
			os.Exit(1)
		}

		//// update status phase
		//var gvr = schema.GroupVersionResource{
		//	Group:    group,
		//	Version:  version,
		//	Resource: plural,
		//}
		//err = clientgo.UpdateCRDStatus(gvr, namespace, name, "phase", "done")
		//if err != nil {
		//	fmt.Printf("Error UpdateCRDStatus err : %v\n", err)
		//	os.Exit(1)
		//}
	},
}

func init() {
	rootCmd.AddCommand(gitCmd)
	gitCmd.Flags().SortFlags = false
	gitCmd.Flags().StringVarP(&gitUrl, "source_url", "s", "", "git url of the resource")
	gitCmd.Flags().StringVarP(&outDir, "output_dir", "o", "/data", "output dir")
	gitCmd.Flags().StringVarP(&group, "group", "g", "cpod.sxwl.ai", "k8s group")
	gitCmd.Flags().StringVarP(&version, "version", "v", "v1", "k8s version")
	gitCmd.Flags().StringVarP(&plural, "plural", "p", "", "k8s plural")
	gitCmd.Flags().StringVarP(&name, "name", "n", "", "k8s name")
	gitCmd.Flags().StringVar(&namespace, "namespace", "cpod", "k8s namespace")
}

func downloadGitRepo(repoURL, outputPath string) error {
	// 检查 git 是否安装
	_, err := exec.LookPath("git")
	if err != nil {
		return fmt.Errorf("git is not installed")
	}

	// 使用 git clone 命令下载仓库
	cmd := exec.Command("git", "clone", repoURL, outputPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error running git clone: %v", err)
	}

	fmt.Printf("Downloaded Git repository from %s and saved to %s\n", repoURL, outputPath)

	return nil
}
