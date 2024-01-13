package testing

import (
	"testing"

	"sxwl/3k/pkg/utils/fs"
)

func TestCreateTmpFile(t *testing.T) {
	p := CreateTmpFile()
	if !fs.Exists(p) {
		t.Errorf("%s should be created, but does not exist", p)
	}

	p = CreateTmpFileWithContent("content")
	if MustReadFile(p) != "content" {
		t.Errorf("p is not empty, %s", p)
	}
}
