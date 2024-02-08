package oss

import (
	"testing"
)

func Test_extractBucketAndObject(t *testing.T) {
	type args struct {
		ossString string
	}
	tests := []struct {
		name       string
		args       args
		wantBucket string
		wantObject string
		wantErr    bool
	}{
		{
			name:       "Valid OSS string",
			args:       args{ossString: "oss://bucket/path/to/object.txt"},
			wantBucket: "bucket",
			wantObject: "path/to/object.txt",
			wantErr:    false,
		},
		{
			name:       "Missing oss:// prefix",
			args:       args{ossString: "bucket/path/to/object.txt"},
			wantBucket: "",
			wantObject: "",
			wantErr:    true,
		},
		{
			name:       "No object in path",
			args:       args{ossString: "oss://bucket"},
			wantBucket: "",
			wantObject: "",
			wantErr:    true,
		},
		{
			name:       "Empty string",
			args:       args{ossString: ""},
			wantBucket: "",
			wantObject: "",
			wantErr:    true,
		},
		{
			name:       "Only oss://",
			args:       args{ossString: "oss://"},
			wantBucket: "",
			wantObject: "",
			wantErr:    true,
		},
		{
			name:       "Valid OSS string with complex path",
			args:       args{ossString: "oss://bucket/another/path/to/object/file.txt"},
			wantBucket: "bucket",
			wantObject: "another/path/to/object/file.txt",
			wantErr:    false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBucket, gotObject, err := ExtractURL(tt.args.ossString)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractURL() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotBucket != tt.wantBucket {
				t.Errorf("ExtractURL() gotBucket = %v, want %v", gotBucket, tt.wantBucket)
			}
			if gotObject != tt.wantObject {
				t.Errorf("ExtractURL() gotObject = %v, want %v", gotObject, tt.wantObject)
			}
		})
	}
}
