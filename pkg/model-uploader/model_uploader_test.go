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

func TestGetUrls(t *testing.T) {
	fileName := "./unittestfile"
	f, err := os.Create(fileName)
	if err != nil {
		t.Error("wrong result", err)
	}
	defer f.Close()
	_, err = f.Write([]byte("line1\nline2\n"))
	if err != nil {
		t.Error("wrong result", err)
	}
	lines, err := GetUrls(fileName)
	if !(err == nil && lines[0] == "line1" && lines[1] == "line2") {
		t.Error("wrong result", err)
	}
	defer os.Remove(fileName)
}
