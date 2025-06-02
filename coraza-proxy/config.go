package main

import (
	"embed"
)

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)



//Custom Directives for Coraza
const customCorazaPath = "/etc/coraza/coraza.conf"

//Custom log file
// The path /var/log/coraza/coraza-audit.log needs write permissions for the user running the app.
// Consider making this configurable via env var or flag.
const loggerFormat = "csv"
const loggerPath ="/var/log/coraza/coraza-audit.csv"





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

