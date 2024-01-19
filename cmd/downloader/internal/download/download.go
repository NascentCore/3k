package download

import (
	"fmt"
	"os"
	"os/exec"
	"sxwl/3k/cmd/downloader/internal/record"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/log"
	"time"
)

type Config struct {
	record.Config
	GitUrl string
	Total  int64
	OutDir string
	Record string
	IsCRD  bool
}

func GitDownload(c Config) error {
	recorder := record.NewRecorder(c.Record, c.Config)

	// check record state
	if err := recorder.Check(); err != nil {
		log.SLogger.Errorf("Error record check %s namespace:%s name:%s err:%s", c.Record, c.Namespace, c.Name, err)
		return err
	}
	log.SLogger.Info("recorder check ok.")

	// empty outDir
	if fs.IsDirExist(c.OutDir) {
		log.SLogger.Infof("empty the outdir:%s", c.OutDir)
		err := fs.RemoveAllFilesInDir(c.OutDir)
		if err != nil {
			log.SLogger.Errorf("Error RemoveAllFilesInDir %s err: %v", c.OutDir, err)
			return err
		}
	}

	// record begin
	err := recorder.Begin()
	if err != nil {
		log.SLogger.Errorf("Error Record begin %s err: %v", c.OutDir, err)
		return err
	}
	log.SLogger.Infof("record crd begin")

	// download repo
	done := make(chan error)
	go func() {
		log.SLogger.Infof("Cloning the repository...")
		err = downloadGitRepo(c.GitUrl, c.OutDir, c.Depth)
		if err != nil {
			log.SLogger.Errorf("Error downloading Git repository err: %v", err)
			_ = recorder.Fail()
		}
		done <- err
	}()

	// Set up a ticker that fires every minute
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	// Loop to check the size every minute or exit when done
	beginTime := time.Now()
	totalSize := "Unknown"
	if c.Total != 0 {
		totalSize = fs.FormatBytes(c.Total)
	}
loop:
	for {
		select {
		case <-ticker.C:
			size, err := fs.GetDirSize(c.OutDir)
			if err != nil {
				log.SLogger.Errorf("Error calculating directory size: %v", err)
				continue
			}
			log.SLogger.Infof("Downloaded size: %s total: %s usedTime: %s",
				fs.FormatBytes(size),
				totalSize,
				time.Since(beginTime).String(),
			)
		case err = <-done:
			if err != nil {
				return err
			}
			size, _ := fs.GetDirSize(c.OutDir)
			log.SLogger.Infof("Downloaded completed size: %s usedTime: %s",
				fs.FormatBytes(size),
				time.Since(beginTime).String(),
			)
			break loop
		}
	}

	// record done
	err = recorder.Complete()
	if err != nil {
		return err
	}
	log.SLogger.Infof("record crd done")

	return nil
}

func downloadGitRepo(repoURL, outputPath string, depth uint) error {
	// 检查 git 是否安装
	_, err := exec.LookPath("git")
	if err != nil {
		return fmt.Errorf("git is not installed")
	}

	// 使用 git clone 命令下载仓库
	var cmd *exec.Cmd
	if depth == 0 {
		cmd = exec.Command("git", "clone", repoURL, outputPath)
	} else {
		cmd = exec.Command("git", "clone", fmt.Sprintf("--depth=%d", depth), repoURL, outputPath)
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("error running git clone: %v", err)
	}

	fmt.Printf("Downloaded Git repository from %s and saved to %s\n", repoURL, outputPath)

	return nil
}
