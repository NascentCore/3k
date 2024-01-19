package fs

import "testing"

func TestFormatBytes(t *testing.T) {
	type args struct {
		bytes int64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test for One KB",
			args: args{bytes: 1024},
			want: "1.00 KB",
		},
		{
			name: "Test for KB",
			args: args{bytes: 1500},
			want: "1.46 KB",
		},
		{
			name: "Test for One MB",
			args: args{bytes: 1048576},
			want: "1.00 MB",
		},
		{
			name: "Test for MB",
			args: args{bytes: 1572864},
			want: "1.50 MB",
		},
		{
			name: "Test for One GB",
			args: args{bytes: 1073741824},
			want: "1.00 GB",
		},
		{
			name: "Test for GB",
			args: args{bytes: 1610612736},
			want: "1.50 GB",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FormatBytes(tt.args.bytes); got != tt.want {
				t.Errorf("FormatBytes() = %v, want %v", got, tt.want)
			}
		})
	}
}
