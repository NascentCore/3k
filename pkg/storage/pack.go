package storage

import (
	"archive/zip"
	"io"
	"os"
	"path"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/log"
)

// 打包Dir下面的所有文件
// excludeFiles排除的文件，通过文件名匹配
// 打包后的文件名是 data.zip
func Pack(dir string, excludeFiles []string) error {
	log.SLogger.Infow("start packing")
	archive, err := os.OpenFile(path.Join(dir, config.PACK_FILE_NAME), os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer archive.Close()
	zipWriter := zip.NewWriter(archive)
	defer zipWriter.Close()
	files, err := FilesToUpload(dir, dir)
	for _, file := range files {
		err = addFile(zipWriter, file)
		if err != nil {
			return err
		}
	}
	log.SLogger.Infow("finish packing")
	return nil
}

func addFile(zipWriter *zip.Writer, file string) error {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w, err := zipWriter.Create(file)
	if err != nil {
		panic(err)
	}
	if _, err := io.Copy(w, f); err != nil {
		panic(err)
	}
	return nil
}
