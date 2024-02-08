package git

import (
	"fmt"
	"os"
	"os/exec"
	"sxwl/3k/cmd/downloader/internal/download/config"
)

type Downloader struct {
	c config.Config
}

func NewDownloader(c config.Config) *Downloader {
	return &Downloader{c}
}

func (d *Downloader) Download() error {
	// 检查 git 是否安装
	_, err := exec.LookPath("git")
	if err != nil {
		return fmt.Errorf("git is not installed")
	}

	// 使用 git clone 命令下载仓库
	var cmd *exec.Cmd
	if d.c.Depth == 0 {
		cmd = exec.Command("git", "clone", d.c.GitUrl, d.c.OutDir)
	} else {
		cmd = exec.Command("git", "clone", fmt.Sprintf("--depth=%d", d.c.Depth), d.c.GitUrl, d.c.OutDir)
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error running git clone: %v", err)
	}

	fmt.Printf("Downloaded Git repository from %s and saved to %s\n", d.c.GitUrl, d.c.OutDir)

	return nil
}
