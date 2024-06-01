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
	CacheAdapter = "adapter"
)

const (
	JobTypePytorch  = "Pytorch"
	JobTypeFinetune = "Finetune" // Rename JobTypeFinetune to JobTypeFineTune
)

const (
	Model                = "model"
	Dataset              = "dataset"
	Adapter              = "adapter"
	OSSPublicModelPath   = "models/public/%s"
	OSSUserModelPath     = "models/%s"
	OSSPublicDatasetPath = "datasets/public/%s"
	OSSUserDatasetPath   = "datasets/%s"
	OSSPublicAdapterPath = "adapters/public/%s"
	OSSUserAdapterPath   = "adapters/%s"
)

const (
	JobTimestampFormat = "20060102-150405"
)
