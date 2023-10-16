package storage

import (
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

// snippet-end:[s3.go.upload_directory.imports]

// directoryIterator represents an iterator of a specified directory
type directoryIterator struct {
	filePaths []string
	bucket    string
	next      struct {
		path string
		f    *os.File
	}
	err error
}

// const exitError = 1

// UploadDirectory uploads the files in a directory to a bucket
// Inputs:
//
//	sess is the current session, which provides configuration for the SDK's service clients
//	bucket is the name of the bucket
//	path is the path to the directory to upload
//
// Output:
//
//	If success, nil
//	Otherwise, an error from the call to UploadWithIterator
func uploadDirectory(sess *session.Session, bucket *string, path *string) error {
	di := NewDirectoryIterator(bucket, path)
	uploader := s3manager.NewUploader(sess)

	err := uploader.UploadWithIterator(aws.BackgroundContext(), di)
	if err != nil {
		return err
	}

	return nil
}

// NewDirectoryIterator builds a new directoryIterator
func NewDirectoryIterator(bucket *string, dir *string) s3manager.BatchUploadIterator {
	var paths []string
	e := filepath.Walk(*dir, func(path string, info os.FileInfo, err error) error {
		if !info.IsDir() {
			paths = append(paths, path)
		}
		return nil
	})
	if e != nil {
		return nil
	}
	return &directoryIterator{
		filePaths: paths,
		bucket:    *bucket,
	}
}

// Next returns whether next file exists
func (di *directoryIterator) Next() bool {
	if len(di.filePaths) == 0 {
		di.next.f = nil
		return false
	}

	f, err := os.Open(di.filePaths[0])
	di.err = err
	di.next.f = f
	di.next.path = di.filePaths[0]
	di.filePaths = di.filePaths[1:]

	return true && di.Err() == nil
}

// Err returns error of directoryIterator
func (di *directoryIterator) Err() error {
	return di.err
}

// UploadObject uploads a file
func (di *directoryIterator) UploadObject() s3manager.BatchUploadObject {
	f := di.next.f
	return s3manager.BatchUploadObject{
		Object: &s3manager.UploadInput{
			Bucket: &di.bucket,
			Key:    &di.next.path,
			Body:   f,
		},
		After: func() error {
			return f.Close()
		},
	}
}

func S3UploadDirectory(directory string) error {
	// TODO: 规划Bucket的使用
	bucket := ""
	sess := session.Must(session.NewSessionWithOptions(session.Options{
		SharedConfigState: session.SharedConfigEnable,
	}))
	err := uploadDirectory(sess, &bucket, &directory)
	return err
}

func S3DownloadFile(url string, filename string) {
	// TODO
}
