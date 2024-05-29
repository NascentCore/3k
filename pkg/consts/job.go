package consts

const (
	JobFail    = "fail"
	JobSuccess = "success"
	JobWorking = "working"
)

const (
	CacheModel   = "model"
	CacheDataSet = "dataset"
	CacheImage   = "image"
)

const (
	JobTypePytorch = "Pytorch"
	// Rename JobTypeFinetune to JobTypeFineTune
	JobTypeFinetune = "FineTune"
)

const (
	Model                = "model"
	Dataset              = "dataset"
	OSSPublicModelPath   = "models/public/%s"
	OSSUserModelPath     = "models/%s"
	OSSPublicDatasetPath = "datasets/public/%s"
	OSSUserDatasetPath   = "datasets/%s"
)

const (
	JobTimestampFormat = "20060102-150405"
)
