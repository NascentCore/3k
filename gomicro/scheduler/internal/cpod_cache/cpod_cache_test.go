package cpod_cache

import "testing"

func TestEncode(t *testing.T) {
	type args struct {
		dataType int64
		dataId   string
	}
	tests := []struct {
		name        string
		args        args
		wantEncoded string
	}{
		{
			name:        "encode dataType and dataId",
			args:        args{dataType: 123, dataId: "abc123"},
			wantEncoded: "123%%%%abc123",
		},
		{
			name:        "encode with empty dataId",
			args:        args{dataType: 456, dataId: ""},
			wantEncoded: "456%%%%",
		},
		{
			name:        "encode with zero dataType",
			args:        args{dataType: 0, dataId: "xyz"},
			wantEncoded: "0%%%%xyz",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotEncoded := Encode(tt.args.dataType, tt.args.dataId); gotEncoded != tt.wantEncoded {
				t.Errorf("Encode() = %v, want %v", gotEncoded, tt.wantEncoded)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	type args struct {
		encoded string
	}
	tests := []struct {
		name         string
		args         args
		wantDataType int64
		wantDataId   string
		wantErr      bool
	}{
		{
			name:         "valid encoded string",
			args:         args{encoded: "123%%%%abc123"},
			wantDataType: 123,
			wantDataId:   "abc123",
			wantErr:      false,
		},
		{
			name:         "invalid format",
			args:         args{encoded: "123abc123"},
			wantDataType: 0,
			wantDataId:   "",
			wantErr:      true,
		},
		{
			name:         "invalid data type",
			args:         args{encoded: "abc%%%%123"},
			wantDataType: 0,
			wantDataId:   "",
			wantErr:      true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDataType, gotDataId, err := Decode(tt.args.encoded)
			if (err != nil) != tt.wantErr {
				t.Errorf("Decode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotDataType != tt.wantDataType {
				t.Errorf("Decode() gotDataType = %v, want %v", gotDataType, tt.wantDataType)
			}
			if gotDataId != tt.wantDataId {
				t.Errorf("Decode() gotDataId = %v, want %v", gotDataId, tt.wantDataId)
			}
		})
	}
}
