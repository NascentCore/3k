package task

type TaskStage int

const (
	TaskStageStart     TaskStage = 0
	TaskStageRunning   TaskStage = 1
	TaskStagePending   TaskStage = 2
	TaskStageFinished  TaskStage = 3
	TaskStageUploading TaskStage = 4
)

type TaskStatus struct {
	TaskID int
	Stage  TaskStage
}

type Task struct {
	Image    string
	DataPath string
	CKPTPath string
}
