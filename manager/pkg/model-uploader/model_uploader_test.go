package modeluploader

import (
	"os"
	"testing"
)

func TestMarkUploadStarted(t *testing.T) {
	fileName := "./unittestfile"
	err := MarkUploadStarted(fileName)
	if err == nil {
		defer os.Remove(fileName)
		d, e := os.ReadFile(fileName)
		if e != nil && string(d) == "let's go" {
			t.Error("read mark file err", e)
		} else if string(d) != "let's go" {
			t.Error("content not match")
		}
	} else {
		t.Error("write file failed", err)
	}
}

func TestCheckUploadStarted(t *testing.T) {
	fileName := "./unittestfile"
	ok, err := CheckUploadStarted(fileName)
	if !(err == nil && !ok) {
		t.Error("wrong result", err)
	}

	if err := MarkUploadStarted(fileName); err == nil {
		defer os.Remove(fileName)
		ok, e := CheckUploadStarted(fileName)
		if !(e == nil && ok) {
			t.Error("wrong result", err)
		}
	}
}
