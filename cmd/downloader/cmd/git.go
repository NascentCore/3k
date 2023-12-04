/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
)

var (
	gitUrl string
	outDir string
)

// gitCmd represents the git command
var gitCmd = &cobra.Command{
	Use:   "git",
	Short: "download data from git url",
	Long:  `download data from git url`,
	Run: func(cmd *cobra.Command, args []string) {
		if gitUrl == "" {
			fmt.Println("please input the git url for downloading")
			return
		}

		err := downloadGitRepo(gitUrl, outDir)
		if err != nil {
			fmt.Printf("Error downloading Git repository: %v\n", err)
			os.Exit(1)
		}
	},
}

func init() {
	rootCmd.AddCommand(gitCmd)
	gitCmd.Flags().SortFlags = false
	gitCmd.Flags().StringVarP(&gitUrl, "source_url", "s", "", "git url of the resource")
	gitCmd.Flags().StringVarP(&outDir, "output_dir", "o", "/data", "output dir")
}

func downloadGitRepo(repoURL, outputPath string) error {
	// 检查 git 是否安装
	_, err := exec.LookPath("git")
	if err != nil {
		return fmt.Errorf("git is not installed")
	}

	// 使用 git clone 命令下载仓库
	cmd := exec.Command("git", "clone", "--depth", "1", repoURL, outputPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error running git clone: %v", err)
	}

	fmt.Printf("Downloaded Git repository from %s and saved to %s\n", repoURL, outputPath)
	return nil
}
