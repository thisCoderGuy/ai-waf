package main

import (
	"embed"
)

// Custom log file
const loggerFormat = "csv"

const (
	logBaseDir = "/var/log/coraza/"

	// A timestamp will be added to the provided filename
	logFileName = "coraza-dataset.csv"

	loggerPath = logBaseDir + logFileName // e.g., "/var/log/coraza/coraza-audit-enum.csv"
)

// Default values for AI verdict and vulnerability type labels.
const DefaultAIVerdictLabel = "unknown"
const DefaultAIVulnerabilityTypeLabel = "unknown_vulnerability"

// File signaling values for AI verdict and vulnerability type labels.
const SHARED_CONFIG_FILE_PATH = "/shared_data/traffic_type.txt"

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)

// Custom Directives for Coraza
const customCorazaPath = "/etc/coraza/coraza.conf"

const wazuhLoggerFormat = "json"
const wazuhLoggerPath = "/var/log/coraza/coraza-audit.json"

// The annotation below embeds the CRS rules in the "owasp-crs-v4/"" folder
//
//go:embed owasp-crs-v4/*
var crs embed.FS

func GetEmbeddedCRSFS() embed.FS {
	return crs
}

func GetAIMicroserviceURL() string {
	return aiMicroserviceURL
}

func GetTargetAppURL() string {
	return targetAppURL
}
