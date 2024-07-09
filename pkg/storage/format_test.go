package storage

import "testing"

func TestBytesToHumanReadable(t *testing.T) {
	type args struct {
		numBytes int64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{name: "Bytes", args: args{numBytes: 1023}, want: "1023.00B"},
		{name: "KiB", args: args{numBytes: 1024}, want: "1.00Ki"},
		{name: "MiB", args: args{numBytes: 1048576}, want: "1.00Mi"},
		{name: "GiB", args: args{numBytes: 1073741824}, want: "1.00Gi"},
		{name: "TiB", args: args{numBytes: 1099511627776}, want: "1.00Ti"},
		{name: "Mixed", args: args{numBytes: 1234567890}, want: "1.15Gi"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BytesToHumanReadable(tt.args.numBytes); got != tt.want {
				t.Errorf("BytesToHumanReadable() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHumanReadableToBytes(t *testing.T) {
	type args struct {
		sizeStr string
	}
	tests := []struct {
		name    string
		args    args
		want    int64
		wantErr bool
	}{
		{name: "Bytes", args: args{sizeStr: "1023B"}, want: 1023, wantErr: false},
		{name: "KiB", args: args{sizeStr: "1Ki"}, want: 1024, wantErr: false},
		{name: "MiB", args: args{sizeStr: "1Mi"}, want: 1048576, wantErr: false},
		{name: "GiB", args: args{sizeStr: "1Gi"}, want: 1073741824, wantErr: false},
		{name: "TiB", args: args{sizeStr: "1Ti"}, want: 1099511627776, wantErr: false},
		{name: "Mixed", args: args{sizeStr: "1.15Gi"}, want: 1234803097, wantErr: false},
		{name: "Invalid", args: args{sizeStr: "10Zi"}, want: 0, wantErr: true},
		{name: "Empty", args: args{sizeStr: ""}, want: 0, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := HumanReadableToBytes(tt.args.sizeStr)
			if (err != nil) != tt.wantErr {
				t.Errorf("HumanReadableToBytes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("HumanReadableToBytes() got = %v, want %v", got, tt.want)
			}
		})
	}
}
