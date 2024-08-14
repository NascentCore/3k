package storage

import (
	"os"
	"path"
	"sxwl/3k/pkg/config"
	"testing"
)

func TestGetUploaded(t *testing.T) {
	dir := "./unittest"
	err := os.Mkdir(dir, os.ModePerm)
	if err != nil {
		t.Error(err)
	}
	err = WriteFile(path.Join(dir, config.FILE_UPLOAD_LOG_FILE), "a\nb")
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(dir)
	res, err := GetUploaded(dir)
	if err != nil {
		t.Error(err)
	}

	if _, ok1 := res["a"]; len(res) != 2 || !ok1 {
		t.Error()
	}
}

func TestFilesToUpload(t *testing.T) {
	outDir := "./outdir"
	err := os.Mkdir(outDir, os.ModePerm)
	if err != nil {
		t.Error(err)
	}
	err = WriteFile(path.Join(outDir, "file1"), ".")
	if err != nil {
		t.Error(err)
	}
	inDir := "./outdir/indir"
	err = os.Mkdir(inDir, os.ModePerm)
	if err != nil {
		t.Error(err)
	}
	err = WriteFile(path.Join(inDir, "file2"), ".")
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(outDir)
	res, err := FilesToUpload(outDir, "pre")
	if err != nil {
		t.Error(err)
	}
	if res[0] != "pre/file1" || res[1] != "pre/indir/file2" {
		t.Error("wrong result")
	}
}
