{{/*
Expand the name of the chart.
*/}}
{{- define "sxcloud.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sxcloud.fullname" -}}
{{- printf "%s-%s" (include "sxcloud.name" .) .Release.Name -}}
{{- end -}}

{{/*
Create common labels.
*/}}
{{- define "sxcloud.labels" -}}
helm.sh/chart: {{ include "sxcloud.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
