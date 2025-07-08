package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"bytes"

	"github.com/corazawaf/coraza/v3/types"
)

// --- CorazaLogger Struct and Methods ---

// LoggerConfig defines the configuration for the logger.
type LoggerConfig struct {
	Format   string // "json" or "csv"
	Filename string
}

type CorazaLogger struct {
	file      *os.File
	logger    *log.Logger
	csvWriter *csv.Writer
	config    LoggerConfig
}

var globalLogger *CorazaLogger

var wazuhLogger *CorazaLogger

// NewCorazaLogger creates a new CorazaLogger based on the provided configuration.
func NewCorazaLogger(config LoggerConfig) (*CorazaLogger, error) {

	dir := filepath.Dir(config.Filename)
	filenameWithExt := filepath.Base(config.Filename)
	baseName := filenameWithExt[:len(filenameWithExt)-len(filepath.Ext(filenameWithExt))]
	extension := filepath.Ext(filenameWithExt)

	timestamp := time.Now().Format("20060102_150405") //In Go, you don't use arbitrary placeholders like YYYY or MM. Instead, use a specific reference time: Mon Jan 2 15:04:05 MST 2006

	//    Example: /var/log/coraza/coraza-audit-benign-20250708_123456.csv
	newFilename := fmt.Sprintf("%s_%s%s", baseName, timestamp, extension)

	finalPath := filepath.Join(dir, newFilename)

	file, err := os.OpenFile(finalPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}

	logger := &CorazaLogger{
		file:   file,
		logger: log.New(file, "", log.LstdFlags),
		config: config,
	}

	if config.Format == "csv" {
		logger.csvWriter = csv.NewWriter(file)
		// Write CSV header only if the file is new/empty
		fileInfo, _ := file.Stat()
		if fileInfo.Size() == 0 {
			if err := logger.writeCSVHeader(); err != nil {
				file.Close()
				return nil, fmt.Errorf("failed to write CSV header: %w", err)
			}
		}
	}

	return logger, nil
}

func SetGlobalLogger(logger *CorazaLogger) {
	globalLogger = logger
}

func GetGlobalLogger() *CorazaLogger {
	return globalLogger
}

func SetWazuhLogger(logger *CorazaLogger) {
	wazuhLogger = logger
}

func GetWazuhLogger() *CorazaLogger {
	return wazuhLogger
}

func (c *CorazaLogger) LogInfo(
	info string,
) {
	c.logger.Println("INFO: " + info)
}

// writeCSVHeader defines and writes the header row for the CSV.
func (c *CorazaLogger) writeCSVHeader() error {
	if c.config.Format != "csv" {
		return nil
	}

	header := []string{
		"Timestamp",
		"TransactionID",
		"ClientIP",
		"ClientPort",
		"ServerIP",
		"ServerPort",
		"RequestMethod",
		"RequestURIPath",
		"RequestURIQuery",
		"RequestProtocol",
		"UserAgent",
		"Referer",
		"AcceptEncoding",
		"ContentType",
		"Accept",
		"Cookie",
		"Connection",
		"OtherHeaders",
		"RequestBody",
		"ResponseStatus",
		"ResponseProtocol",
		"ResponseHeaders",
		"ResponseBody",
		"WAFInterrupted",
		"InterruptionRuleID",
		"InterruptionStatus",
		"InterruptionAction",
		"MatchedRulesCount",
		"MatchedRulesIDs",
		"MatchedRulesMessages",
		"MatchedRulesTags",
		"AIScore",
		"AIVerdict",
		"AIVerdictLabel",           // New field for AI verdict label
		"AIVulnerabilityTypeLabel", // New field for AI vulnerability type label
		// Calculated fields
		"RequestLength",
		"PathLength",
		"QueryLength",
		"BodyLength",
		"NumParams",
		"LongestParamValueLength",
		"NumSpecialChars",
		"NumEncodedChars",
		"NumDoubleEncodedChars",
		"RatioSpecialCharsToTotalChars",
		"NumKeywords",
		"NumUniqueChars",
		"IsJSONBody",
		"IsXMLBody",
		"IsMultipartForm",
		"HasNumericPathSegment",
		"DepthOfPath",
		"UsesBase64Encoding",
		"UsesHexEncoding",
		"HasMultipleContentLength",
		"InvalidHTTPVersion",
		"NumHTTPHeaders",
		"TimeOfDayHour",
		"TimeOfDayDayOfWeek",
	}
	if err := c.csvWriter.Write(header); err != nil {
		return err
	}
	c.csvWriter.Flush()
	return c.csvWriter.Error()
}

// LogTransaction dispatches logging to either JSON or CSV format.
func (c *CorazaLogger) LogTransaction(
	tx types.Transaction,
	req *http.Request,
	res *http.Response,
	aiScore float64,
	aiVerdict string,
) {

	// Initialize new fields with default values from the same package (main)
	// No explicit import needed, just use the constant name directly.
	aiVerdictLabel := DefaultAIVerdictLabel
	aiVulnerabilityTypeLabel := DefaultAIVulnerabilityTypeLabel

	if c.config.Format == "json" {
		c.logTransactionJSON(tx, req, res, aiScore, aiVerdict, aiVerdictLabel, aiVulnerabilityTypeLabel)
	} else if c.config.Format == "csv" {
		c.logTransactionCSV(tx, req, res, aiScore, aiVerdict, aiVerdictLabel, aiVulnerabilityTypeLabel)
	} else {
		c.logger.Printf("Unknown log format: %s", c.config.Format)
	}
}

// CorazaLogEntry defines the structure for JSON log entries.
type CorazaLogEntry struct {
	Timestamp     time.Time            `json:"timestamp"`
	TransactionID string               `json:"transaction_id"`
	ClientIP      string               `json:"client_ip"`
	ClientPort    int                  `json:"client_port"`
	ServerIP      string               `json:"server_ip"`
	ServerPort    int                  `json:"server_port"`
	Request       RequestLogData       `json:"request"`
	Response      *ResponseLogData     `json:"response,omitempty"`
	WAFProcessing WAFProcessingLogData `json:"waf_processing"`
	Calculated    CalculatedFields     `json:"calculated_fields"`
}

// RequestLogData holds details about the HTTP request.
type RequestLogData struct {
	Method   string        `json:"method"`
	URI      string        `json:"uri"`
	Path     string        `json:"path"`
	Query    string        `json:"query"`
	Protocol string        `json:"protocol"`
	Headers  ParsedHeaders `json:"headers"`
	Body     string        `json:"body"`
}

// ResponseLogData holds details about the HTTP response.
type ResponseLogData struct {
	StatusCode int         `json:"status_code"`
	Protocol   string      `json:"protocol"`
	Headers    http.Header `json:"headers"`
	Body       string      `json:"body"` // Response body might be empty if not captured
}

// WAFProcessingLogData holds details about WAF processing and AI analysis.
type WAFProcessingLogData struct {
	Interrupted              bool                 `json:"interrupted"`
	InterruptionDetails      *InterruptionLogData `json:"interruption_details,omitempty"`
	MatchedRules             []MatchedRuleLogData `json:"matched_rules"`
	AIScore                  float64              `json:"ai_score"`
	AIVerdict                string               `json:"ai_verdict"`
	AIVerdictLabel           string               `json:"ai_verdict_label"`            // New field
	AIVulnerabilityTypeLabel string               `json:"ai_vulnerability_type_label"` // New field
}

// InterruptionLogData holds details about a WAF interruption.
type InterruptionLogData struct {
	RuleID string `json:"rule_id"`
	Status int    `json:"status"`
	Action string `json:"action"`
}

// MatchedRuleLogData holds details about a matched WAF rule.
type MatchedRuleLogData struct {
	RuleID  string   `json:"rule_id"`
	Message string   `json:"message"`
	Tags    []string `json:"tags"`
}

// ParsedHeaders holds categorized request headers.
type ParsedHeaders struct {
	UserAgent      string              `json:"user_agent,omitempty"`
	Referer        string              `json:"referer,omitempty"`
	AcceptEncoding string              `json:"accept_encoding,omitempty"`
	ContentType    string              `json:"content_type,omitempty"`
	Accept         string              `json:"accept,omitempty"`
	Cookie         string              `json:"cookie,omitempty"`
	Connection     string              `json:"connection,omitempty"`
	OtherHeaders   map[string][]string `json:"other_headers,omitempty"`
}

// CalculatedFields holds various calculated metrics for analysis.
type CalculatedFields struct {
	RequestLength                 int     `json:"request_length"`
	PathLength                    int     `json:"path_length"`
	QueryLength                   int     `json:"query_ength"`
	BodyLength                    int     `json:"body_length"`
	NumParams                     int     `json:"num_params"`
	LongestParamValueLength       int     `json:"longest_param_value_length"`
	NumSpecialChars               int     `json:"num_special_chars"`
	NumEncodedChars               int     `json:"num_encoded_chars"`
	NumDoubleEncodedChars         int     `json:"num_double_encoded_chars"`
	RatioSpecialCharsToTotalChars float64 `json:"ratio_special_chars_to_total_chars"`
	NumKeywords                   int     `json:"num_keywords"`
	NumUniqueChars                int     `json:"num_unique_chars"`
	IsJSONBody                    bool    `json:"is_json_body"`
	IsXMLBody                     bool    `json:"is_xml_body"`
	IsMultipartForm               bool    `json:"is_multipart_form"`
	HasNumericPathSegment         bool    `json:"has_numeric_path_segment"`
	DepthOfPath                   int     `json:"depth_of_path"`
	UsesBase64Encoding            bool    `json:"uses_base64_encoding"`
	UsesHexEncoding               bool    `json:"uses_hex_encoding"`
	HasMultipleContentLength      bool    `json:"has_multiple_content_length"`
	InvalidHTTPVersion            bool    `json:"invalid_http_version"`
	NumHTTPHeaders                int     `json:"num_http_headers"`
	TimeOfDayHour                 int     `json:"time_of_day_hour"`
	TimeOfDayDayOfWeek            string  `json:"time_of_day_day_of_week"`
}

// logTransactionJSON logs transaction details in JSON format.
func (c *CorazaLogger) logTransactionJSON(
	tx types.Transaction,
	req *http.Request,
	res *http.Response,
	aiScore float64,
	aiVerdict string,
	aiVerdictLabel string, // New parameter
	aiVulnerabilityTypeLabel string, // New parameter
) {
	clientIP, clientPortStr, _ := net.SplitHostPort(req.RemoteAddr)
	clientPort, _ := strconv.Atoi(clientPortStr)

	serverHost := req.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		serverHost += ":80"
	}
	serverIP, serverPortStr, _ := net.SplitHostPort(serverHost)
	serverPort, _ := strconv.Atoi(serverPortStr)

	var reqBodyBytes []byte
	if req.Body != nil {
		var err error
		reqBodyBytes, err = ioutil.ReadAll(req.Body)
		if err != nil {
			c.logger.Printf("Failed to read request body: %v", err)
		}
		req.Body = ioutil.NopCloser(bytes.NewBuffer(reqBodyBytes))
	}

	uriPath, uriQuery := splitRequestURI(req.URL)
	parsedHeaders := parseRequestHeaders(req.Header)

	entry := CorazaLogEntry{
		Timestamp:     time.Now(),
		TransactionID: tx.ID(),
		ClientIP:      clientIP,
		ClientPort:    clientPort,
		ServerIP:      serverIP,
		ServerPort:    serverPort,
		Request: RequestLogData{
			Method:   req.Method,
			URI:      req.URL.String(),
			Path:     uriPath,
			Query:    uriQuery,
			Protocol: req.Proto,
			Headers:  parsedHeaders,
			Body:     string(reqBodyBytes),
		},
	}

	var resBodyBytes []byte // Initialize resBodyBytes
	if res != nil {
		entry.Response = &ResponseLogData{
			StatusCode: res.StatusCode,
			Protocol:   res.Proto,
			Headers:    res.Header,
			Body:       string(resBodyBytes), // Populate if response body is captured
		}
	}

	entry.WAFProcessing.Interrupted = tx.IsInterrupted()
	if it := tx.Interruption(); it != nil {
		entry.WAFProcessing.InterruptionDetails = &InterruptionLogData{
			RuleID: fmt.Sprintf("%d", it.RuleID),
			Status: it.Status,
			Action: it.Action,
		}
	}

	matchedRules := []MatchedRuleLogData{}
	for _, mr := range tx.MatchedRules() {
		matchedRules = append(matchedRules, MatchedRuleLogData{
			RuleID:  fmt.Sprintf("%d", mr.Rule().ID()),
			Message: mr.Message(),
			Tags:    mr.Rule().Tags(),
		})
	}
	entry.WAFProcessing.MatchedRules = matchedRules
	entry.WAFProcessing.AIScore = aiScore
	entry.WAFProcessing.AIVerdict = aiVerdict
	entry.WAFProcessing.AIVerdictLabel = aiVerdictLabel                     // Set new field
	entry.WAFProcessing.AIVulnerabilityTypeLabel = aiVulnerabilityTypeLabel // Set new field

	entry.Calculated = calculateFields(req, string(reqBodyBytes), uriPath, uriQuery)

	jsonData, err := json.MarshalIndent(entry, "", " ") // Use MarshalIndent for pretty printing
	if err != nil {
		c.logger.Printf("Failed to marshal log data: %v", err)
		return
	}
	c.logger.Println(string(jsonData))
}

// logTransactionCSV logs transaction details in CSV format.
func (c *CorazaLogger) logTransactionCSV(
	tx types.Transaction,
	req *http.Request,
	res *http.Response,
	aiScore float64,
	aiVerdict string,
	aiVerdictLabel string, // New parameter
	aiVulnerabilityTypeLabel string, // New parameter
) {
	clientIP, clientPortStr, _ := net.SplitHostPort(req.RemoteAddr)
	clientPort, _ := strconv.Atoi(clientPortStr)

	serverHost := req.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		serverHost += ":80"
	}
	serverIP, serverPortStr, _ := net.SplitHostPort(serverHost)
	serverPort, _ := strconv.Atoi(serverPortStr)

	var reqBodyBytes []byte
	if req.Body != nil {
		var err error
		reqBodyBytes, err = ioutil.ReadAll(req.Body)
		if err != nil {
			c.logger.Printf("Failed to read request body for logging: %v", err)
		}
		req.Body = ioutil.NopCloser(bytes.NewBuffer(reqBodyBytes))
	}

	uriPath, uriQuery := splitRequestURI(req.URL)
	parsedHeaders := parseRequestHeaders(req.Header)

	interrupted := tx.IsInterrupted()
	var interruptionRuleID, interruptionAction string
	var interruptionStatus int
	if it := tx.Interruption(); it != nil {
		interruptionRuleID = fmt.Sprintf("%d", it.RuleID)
		interruptionStatus = it.Status
		interruptionAction = it.Action
	}

	var matchedRuleIDs, matchedRuleMessages, matchedRuleTags []string
	for _, mr := range tx.MatchedRules() {
		matchedRuleIDs = append(matchedRuleIDs, fmt.Sprintf("%d", mr.Rule().ID()))
		matchedRuleMessages = append(matchedRuleMessages, mr.Message())
		matchedRuleTags = append(matchedRuleTags, strings.Join(mr.Rule().Tags(), ";"))
	}

	calculated := calculateFields(req, string(reqBodyBytes), uriPath, uriQuery)

	record := []string{
		time.Now().Format(time.RFC3339),
		tx.ID(),
		clientIP,
		strconv.Itoa(clientPort),
		serverIP,
		strconv.Itoa(serverPort),
		req.Method,
		uriPath,
		uriQuery,
		req.Proto,
		parsedHeaders.UserAgent,
		parsedHeaders.Referer,
		parsedHeaders.AcceptEncoding,
		parsedHeaders.ContentType,
		parsedHeaders.Accept,
		parsedHeaders.Cookie,
		parsedHeaders.Connection,
		flattenOtherHeaders(parsedHeaders.OtherHeaders),
		string(reqBodyBytes),
	}

	if res != nil {
		record = append(record, strconv.Itoa(res.StatusCode))
		record = append(record, res.Proto)
		record = append(record, flattenHeaders(res.Header))
		record = append(record, "") // Response body is not captured in this flow currently
	} else {
		record = append(record, "", "", "", "")
	}

	record = append(record,
		strconv.FormatBool(interrupted),
		interruptionRuleID,
		strconv.Itoa(interruptionStatus),
		interruptionAction,
		strconv.Itoa(len(tx.MatchedRules())),
		strings.Join(matchedRuleIDs, ";"),
		strings.Join(matchedRuleMessages, ";"),
		strings.Join(matchedRuleTags, ";"),
		fmt.Sprintf("%.2f", aiScore),
		aiVerdict,
		aiVerdictLabel,           // Append new field
		aiVulnerabilityTypeLabel, // Append new field
		// Append calculated fields
		strconv.Itoa(calculated.RequestLength),
		strconv.Itoa(calculated.PathLength),
		strconv.Itoa(calculated.QueryLength),
		strconv.Itoa(calculated.BodyLength),
		strconv.Itoa(calculated.NumParams),
		strconv.Itoa(calculated.LongestParamValueLength),
		strconv.Itoa(calculated.NumSpecialChars),
		strconv.Itoa(calculated.NumEncodedChars),
		strconv.Itoa(calculated.NumDoubleEncodedChars),
		fmt.Sprintf("%.4f", calculated.RatioSpecialCharsToTotalChars),
		strconv.Itoa(calculated.NumKeywords),
		strconv.Itoa(calculated.NumUniqueChars),
		strconv.FormatBool(calculated.IsJSONBody),
		strconv.FormatBool(calculated.IsXMLBody),
		strconv.FormatBool(calculated.IsMultipartForm),
		strconv.FormatBool(calculated.HasNumericPathSegment),
		strconv.Itoa(calculated.DepthOfPath),
		strconv.FormatBool(calculated.UsesBase64Encoding),
		strconv.FormatBool(calculated.UsesHexEncoding),
		strconv.FormatBool(calculated.HasMultipleContentLength),
		strconv.FormatBool(calculated.InvalidHTTPVersion),
		strconv.Itoa(calculated.NumHTTPHeaders),
		strconv.Itoa(calculated.TimeOfDayHour),
		calculated.TimeOfDayDayOfWeek,
	)

	if err := c.csvWriter.Write(record); err != nil {
		c.logger.Printf("Failed to write CSV record: %v", err)
	}
	c.csvWriter.Flush()
	if err := c.csvWriter.Error(); err != nil {
		c.logger.Printf("CSV writer error: %v", err)
	}
}

// splitRequestURI separates the request URI into path and query.
func splitRequestURI(u *url.URL) (path, query string) {
	path = u.Path
	if u.RawQuery != "" {
		query = u.RawQuery
	}
	return path, query
}

// parseRequestHeaders categorizes and concatenates request headers.
func parseRequestHeaders(headers http.Header) ParsedHeaders {
	parsed := ParsedHeaders{
		OtherHeaders: make(map[string][]string),
	}

	for name, values := range headers {
		headerName := http.CanonicalHeaderKey(name)
		concatenatedValues := strings.Join(values, ", ")

		switch headerName {
		case "User-Agent":
			parsed.UserAgent = concatenatedValues
		case "Referer":
			parsed.Referer = concatenatedValues
		case "Accept-Encoding":
			parsed.AcceptEncoding = concatenatedValues
		case "Content-Type":
			parsed.ContentType = concatenatedValues
		case "Accept":
			parsed.Accept = concatenatedValues
		case "Cookie":
			parsed.Cookie = concatenatedValues
		case "Connection":
			parsed.Connection = concatenatedValues
		default:
			parsed.OtherHeaders[name] = values // Store original name to preserve case for other headers
		}
	}
	return parsed
}

// flattenHeaders flattens all headers into a single string for CSV.
func flattenHeaders(headers http.Header) string {
	var sb strings.Builder
	first := true
	for name, values := range headers {
		if !first {
			sb.WriteString("; ")
		}
		sb.WriteString(name)
		sb.WriteString(":")
		sb.WriteString(strings.Join(values, ","))
		first = false
	}
	return sb.String()
}

// flattenOtherHeaders flattens the 'OtherHeaders' map into a single string for CSV.
func flattenOtherHeaders(headers map[string][]string) string {
	var sb strings.Builder
	first := true
	for name, values := range headers {
		if !first {
			sb.WriteString("; ")
		}
		sb.WriteString(name)
		sb.WriteString(":")
		sb.WriteString(strings.Join(values, ","))
		first = false
	}
	return sb.String()
}

// Close flushes any pending writes and closes the file.
func (c *CorazaLogger) Close() error {
	if c.config.Format == "csv" {
		c.csvWriter.Flush()
		if err := c.csvWriter.Error(); err != nil {
			c.logger.Printf("Error flushing CSV writer before closing: %v", err)
		}
	}
	return c.file.Close()
}

// --- New Calculated Fields Functions ---

// calculateFields calculates various metrics from the request.
func calculateFields(req *http.Request, body string, path string, query string) CalculatedFields {
	calc := CalculatedFields{}

	rawRequest := buildRawRequest(req, body)
	calc.RequestLength = len(rawRequest)

	calc.PathLength = len(path)
	calc.QueryLength = len(query)
	calc.BodyLength = len(body)

	calc.NumParams, calc.LongestParamValueLength = countParamsAndLongestValue(req.URL.RawQuery, body, req.Header.Get("Content-Type"))

	allContent := path + query + body
	calc.NumSpecialChars = countSpecialChars(allContent)
	calc.NumEncodedChars, calc.NumDoubleEncodedChars = countEncodedChars(allContent)
	if len(allContent) > 0 {
		calc.RatioSpecialCharsToTotalChars = float64(calc.NumSpecialChars) / float64(len(allContent))
	}

	calc.NumKeywords = countKeywords(allContent)
	calc.NumUniqueChars = countUniqueChars(allContent)

	calc.IsJSONBody = isJSON(body)
	calc.IsXMLBody = isXML(body)
	calc.IsMultipartForm = strings.HasPrefix(req.Header.Get("Content-Type"), "multipart/form-data")
	calc.HasNumericPathSegment = hasNumericPathSegment(path)
	calc.DepthOfPath = countPathDepth(path)

	calc.UsesBase64Encoding = usesBase64Encoding(allContent)
	calc.UsesHexEncoding = usesHexEncoding(allContent)

	calc.HasMultipleContentLength = len(req.Header["Content-Length"]) > 1
	calc.InvalidHTTPVersion = !isValidHTTPVersion(req.Proto)
	calc.NumHTTPHeaders = len(req.Header)

	now := time.Now()
	calc.TimeOfDayHour = now.Hour()
	calc.TimeOfDayDayOfWeek = now.Weekday().String()

	return calc
}

// buildRawRequest reconstructs a simplified raw HTTP request for length calculation.
func buildRawRequest(req *http.Request, body string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s %s %s\r\n", req.Method, req.RequestURI, req.Proto))
	for name, values := range req.Header {
		for _, value := range values {
			sb.WriteString(fmt.Sprintf("%s: %s\r\n", name, value))
		}
	}
	sb.WriteString("\r\n")
	sb.WriteString(body)
	return sb.String()
}

// countParamsAndLongestValue counts parameters and finds the longest parameter value.
func countParamsAndLongestValue(queryString, body string, contentType string) (int, int) {
	numParams := 0
	longestParamValueLength := 0

	// Parse query string parameters
	if queryString != "" {
		m, err := url.ParseQuery(queryString)
		if err == nil {
			for _, values := range m {
				for _, v := range values {
					numParams++
					if len(v) > longestParamValueLength {
						longestParamValueLength = len(v)
					}
				}
			}
		}
	}

	// Parse body parameters for form-urlencoded
	if strings.Contains(contentType, "application/x-www-form-urlencoded") {
		m, err := url.ParseQuery(body)
		if err == nil {
			for _, values := range m {
				for _, v := range values {
					numParams++
					if len(v) > longestParamValueLength {
						longestParamValueLength = len(v)
					}
				}
			}
		}
	}

	// For JSON/XML bodies, consider top-level keys/values or string values within.
	// This is a simplified approach; a full parser would be more accurate.
	if isJSON(body) {
		var jsonMap map[string]interface{}
		if err := json.Unmarshal([]byte(body), &jsonMap); err == nil {
			for _, v := range jsonMap {
				numParams++ // Each top-level key is a "param"
				if strVal, ok := v.(string); ok {
					if len(strVal) > longestParamValueLength {
						longestParamValueLength = len(strVal)
					}
				}
			}
		}
	} else if isXML(body) {
		// Basic XML parsing for parameters (highly simplified)
		// This would typically require an XML parser to be accurate.
		// For now, we'll just count any string values that look like they could be parameters.
		r := regexp.MustCompile(`>([^<]+)<`) // find content between tags
		matches := r.FindAllStringSubmatch(body, -1)
		for _, match := range matches {
			if len(match) > 1 {
				numParams++
				if len(match[1]) > longestParamValueLength {
					longestParamValueLength = len(match[1])
				}
			}
		}
	}

	return numParams, longestParamValueLength
}

// countSpecialChars counts specific special characters.
func countSpecialChars(s string) int {
	specialChars := `!@#$%^&*()_+{}[]|\:;"'<>,.?/~` // Added more common special characters
	count := 0
	for _, r := range s {
		if strings.ContainsRune(specialChars, r) {
			count++
		}
	}
	return count
}

// countEncodedChars counts URL-encoded characters (%xx).
func countEncodedChars(s string) (int, int) {
	encodedCount := 0
	doubleEncodedCount := 0
	re := regexp.MustCompile(`%[0-9a-fA-F]{2}`)
	matches := re.FindAllString(s, -1)
	encodedCount = len(matches)

	// A simple heuristic for double encoding: look for %25 followed by %xx
	reDouble := regexp.MustCompile(`%25[0-9a-fA-F]{2}`)
	doubleEncodedCount = len(reDouble.FindAllString(s, -1))

	return encodedCount, doubleEncodedCount
}

// countKeywords counts occurrences of suspicious keywords.
func countKeywords(s string) int {
	keywords := []string{
		"union", "select", "sleep", "script", "alert", "etc/passwd",
		"cmd.exe", "exec", "system", "wget", "nc", "rm -rf", "insert",
		"drop", "alter", "delete from", "xp_cmdshell", "benchmark", "sleep",
		"information_schema", "/etc/passwd", "file_get_contents", "passthru",
		"shell_exec", "system", "base64_decode", "php://filter", "data://",
		"input.php", "load_file", "outfile", "dumpfile", "load data",
		"into outfile", "into dumpfile", "union all", "union select",
		"cast(", "convert(", "declare @", "nchar(", "varchar(", "nvarchar(",
		"substring(", "mid(", "concat(", "char(", "chr(", "convert(", "cast(",
		"schema_name()", "table_name()", "column_name()", "database()", "version()",
		"user()", "current_user()", "session_user()", "system_user()", "@@version",
		"@@hostname", "load_file", "select pg_sleep", "select benCHMARK",
		"document.cookie", "window.location", "eval(", "prompt(", "confirm(",
		"alert(", "javascript:", "vbscript:", "<script>", "</script>", "<iframe>",
		"</iframe>", "<embed>", "</embed>", "<object>", "</object>", "<svg>",
		"</svg>", "<img src=x onerror=", "onload=", "onerror=", "onmouseover=",
		"onfocus=", "autofocus", "background-image:url(", "expression(",
		"xss", "cross-site scripting", "sql injection", "remote code execution",
		"local file inclusion", "remote file inclusion",
	}
	count := 0
	lowerS := strings.ToLower(s)
	for _, keyword := range keywords {
		count += strings.Count(lowerS, strings.ToLower(keyword))
	}
	return count
}

// countUniqueChars counts the number of unique characters in a string.
func countUniqueChars(s string) int {
	seen := make(map[rune]bool)
	for _, r := range s {
		seen[r] = true
	}
	return len(seen)
}

// isJSON checks if a string is a valid JSON.
func isJSON(s string) bool {
	var js json.RawMessage
	return json.Unmarshal([]byte(s), &js) == nil
}

// isXML checks if a string is a valid XML.
func isXML(s string) bool {
	// A simple check for XML by looking for a root element.
	// A more robust solution would involve a full XML parser.
	return strings.HasPrefix(strings.TrimSpace(s), "<") && strings.HasSuffix(strings.TrimSpace(s), ">") &&
		(strings.Contains(s, "<?xml") || strings.Contains(s, "xmlns"))
}

// hasNumericPathSegment checks if any path segment contains only digits.
func hasNumericPathSegment(path string) bool {
	segments := strings.Split(path, "/")
	for _, segment := range segments {
		if segment != "" && regexp.MustCompile(`^\d+$`).MatchString(segment) {
			return true
		}
	}
	return false
}

// countPathDepth counts the number of '/' in the path.
func countPathDepth(path string) int {
	if path == "/" || path == "" {
		return 0
	}
	return strings.Count(strings.Trim(path, "/"), "/") + 1
}

// usesBase64Encoding detects common base64 patterns.
func usesBase64Encoding(s string) bool {
	// Base64 regex: characters A-Z, a-z, 0-9, +, /, =
	// Minimum length for a meaningful base64 string (e.g., "AA==" for one byte)
	// Must be a multiple of 4 characters, padded with '='
	re := regexp.MustCompile(`(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?`)
	return re.MatchString(s) && len(s)%4 == 0 && utf8.ValidString(s) // Ensure it's valid UTF-8
}

// usesHexEncoding detects common hex patterns.
func usesHexEncoding(s string) bool {
	// Hex pattern: sequences of two hex characters
	re := regexp.MustCompile(`(?:%[0-9a-fA-F]{2})+`) // e.g., %20%41
	if re.MatchString(s) {
		return true
	}
	// Another common hex pattern, e.g., \x41\x42
	re = regexp.MustCompile(`(?:\\x[0-9a-fA-F]{2})+`)
	return re.MatchString(s)
}

// isValidHTTPVersion checks if the HTTP protocol version is valid.
func isValidHTTPVersion(proto string) bool {
	return proto == "HTTP/1.0" || proto == "HTTP/1.1" || proto == "HTTP/2.0"
}
