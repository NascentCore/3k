package testing

import (
	"os"
	"path"
	"log"
	"math/rand"
	"time"
)

// Initialize random seed for randStr()
func init() {
	rand.New(rand.NewSource(time.Now().UnixNano()))
}

// createTmpDir returns a path to a newly created temporary directory.
func createTmpDir() string {
	prefix := "sxwl-test-"
	dir := path.Join(os.TempDir(), prefix+randStr(10))
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		log.Fatalf("Failed to create tmp dir '%s', error: %v", dir, err)
	}
	return dir
}

func randStr(n int) string {
	var letters = []rune("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

// getTmpFilePath returns a random file path under temp directory.
// The file is not created.
func getTmpFilePath() string {
	return path.Join(createTmpDir(), randStr(10))
}

// CreateTmpFile returns a path to a file under the temporary directory.
func CreateTmpFile() string {
	f := getTmpFilePath()
	openedFile, err := os.Create(f)
	if err != nil {
		log.Fatalf("While creating temp file at '%s', failed to create the file, error: %v", f, err)
	}
	err = openedFile.Close()
	if err != nil {
		log.Fatalf("While creating temp file at '%s', failed to close the file after creation, error: %v", f, err)
	}
	return f
}

func CreateTmpFileWithContent(content string) string {
	f := getTmpFilePath()
	const defaultPerf = 0o644
	err := os.WriteFile(f, []byte(content), defaultPerf)
	if err != nil {
		log.Fatalf("Failed to write to file '%s', error: %v", f, err)
	}
	return f
}

func MustReadFile(f string) string {
	content, err := os.ReadFile(f)
	if err != nil {
		log.Fatalf("Failed to read file '%s', error: %v", f, err)
	}
	return string(content)
}
