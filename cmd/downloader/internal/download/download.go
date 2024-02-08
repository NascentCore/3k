package download

import (
	"sxwl/3k/cmd/downloader/internal/download/config"
	"sxwl/3k/cmd/downloader/internal/record"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/log"
	"time"
)

func Download(c config.Config, downloader Downloader, recorder record.Recorder) error {
	// check record state
	if err := recorder.Check(); err != nil {
		log.SLogger.Errorf("Error record check %s namespace:%s name:%s err:%s", c.Record, c.Namespace, c.Name, err)
		return err
	}
	log.SLogger.Infof("Recorder %s check ok.", recorder.Name())

	// empty outDir
	if fs.IsDirExist(c.OutDir) {
		log.SLogger.Infof("Empty the outdir:%s", c.OutDir)
		err := fs.RemoveAllFilesInDir(c.OutDir)
		if err != nil {
			log.SLogger.Errorf("Error RemoveAllFilesInDir %s err: %v", c.OutDir, err)
			return err
		}
	}

	// record begin
	err := recorder.Begin()
	if err != nil {
		log.SLogger.Errorf("Record %s begin err: %v", recorder.Name(), err)
		return err
	}
	log.SLogger.Infof("Record %s begin", recorder.Name())

	// download repo
	done := make(chan error)
	go func() {
		log.SLogger.Infof("Begin downloading")
		err = downloader.Download()
		if err != nil {
			log.SLogger.Errorf("Downloading err: %v", err)
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
	log.SLogger.Infof("Record %s done", recorder.Name())

	return nil
}
