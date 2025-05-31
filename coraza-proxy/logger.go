package main

import (
	"encoding/csv" 
	"encoding/json"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"io/ioutil"
	"time"
	"bytes"
	"strings"
	"fmt"
	"github.com/corazawaf/coraza/v3/types"
)


// --- CorazaLogger Struct and Methods ---
type CorazaLogger struct {
	file   *os.File
	logger *log.Logger
	csvWriter *csv.Writer
}

var globalLogger *CorazaLogger

func NewCorazaLogger(filename string) (*CorazaLogger, error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}

	csvWriter := csv.NewWriter(file)

	logger := &CorazaLogger{
		file:      file,
		logger:    log.New(file, "", log.LstdFlags), // You can still use this for non-CSV specific logs if needed
		csvWriter: csvWriter,
	}



	// Write CSV header only if the file is new/empty
	fileInfo, _ := file.Stat()
	if fileInfo.Size() == 0 {
		if err := logger.writeCSVHeader(); err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to write CSV header: %w", err)
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

func (c *CorazaLogger) LogInfo(
	info string, 
) {
	c.logger.Println("INFO: "+ info)
}

// writeCSVHeader defines and writes the header row for the CSV.
func (c *CorazaLogger) writeCSVHeader() error {
	header := []string{
		"Timestamp",
		"TransactionID",
		"ClientIP",
		"ClientPort",
		"ServerIP",
		"ServerPort",
		"RequestMethod",
		"RequestURI",
		"RequestProtocol",
		"RequestHeaders", // Flattened
		"RequestBody",    // Flattened
		"ResponseStatus",
		"ResponseProtocol",
		"ResponseHeaders", // Flattened
		"ResponseBody",    // Flattened
		"WAFInterrupted",
		"InterruptionRuleID",
		"InterruptionStatus",
		"InterruptionAction",
		"MatchedRulesCount",
		"MatchedRulesIDs",     // Flattened
		"MatchedRulesMessages", // Flattened
		"MatchedRulesTags",    // Flattened
		"AIScore",
		"AIVerdict",
	}
	if err := c.csvWriter.Write(header); err != nil {
		return err
	}
	c.csvWriter.Flush() // Ensure header is written to file
	return c.csvWriter.Error()
}


func (c *CorazaLogger) LogTransactionJSON(
	tx types.Transaction,
	req *http.Request,
	res *http.Response,
	aiScore float64, // Directly pass AI score
	aiVerdict string, // Directly pass AI verdict
) {
	clientIP, clientPortStr, err := net.SplitHostPort(req.RemoteAddr)
	clientPort, _ := strconv.Atoi(clientPortStr)
	
	serverHost := req.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		serverHost += ":80"
	}
	serverIP, serverPortStr, err := net.SplitHostPort(serverHost)
	serverPort, _ := strconv.Atoi(serverPortStr)

	entry := CorazaLogEntry{
		Timestamp:     time.Now(),
		TransactionID: tx.ID(),
		ClientIP:      clientIP,
		ClientPort:    clientPort,
		ServerIP:      serverIP,
		ServerPort:    serverPort,
	}

	var bodyBytes []byte 
	if req.Body != nil {
		var err error
		// Read the entire request body.
		bodyBytes, err = ioutil.ReadAll(req.Body)
		if err != nil {
			//log.Printf("Failed to read request body: %v", err)
			return // Exit if body cannot be read
		}
		// Restore the body for subsequent handlers (AI service and reverse proxy)
		// as `ioutil.ReadAll` consumes the original `r.Body`.
		req.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
	}


	// Request Data
	entry.Request = RequestLogData{
		Method:   req.Method,
		URI:      req.URL.String(),
		Protocol: req.Proto,
		Headers:  req.Header, // Raw headers
		Body:      string(bodyBytes),
	}

	
	/*if res.Body != nil {
			bodyBytes, err = ioutil.ReadAll(res.Body)
			if err != nil {
				//log.Printf("Failed to read response body: %v", err)
				return
			}
			res.Body.Close() // Close the original body to prevent resource leaks
			// Restore the body for the client by creating a new io.ReadCloser from the bytes.
			res.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))

			// Process the response body with Coraza.
			tx.ProcessResponseBody()
		}
			*/

	// Response Data
	if res != nil {
		entry.Response = ResponseLogData{
			StatusCode: res.StatusCode,
			Protocol:   res.Proto,
			Headers:    res.Header, // Raw headers
			Body:       string(bodyBytes),
		}
	}

	// WAF Processing Data
	entry.WAFProcessing.Interrupted = tx.IsInterrupted()
	if it := tx.Interruption(); it != nil {
		entry.WAFProcessing.InterruptionDetails = &InterruptionLogData{
			RuleID: fmt.Sprintf("%d", it.RuleID), // RuleID is int in types.Interruption
			Status: it.Status,
			Action: it.Action,
		}
	}

	// Matched Rules
	matchedRules := []MatchedRuleLogData{}
	for _, mr := range tx.MatchedRules() {
		matchedRules = append(matchedRules, MatchedRuleLogData{
			RuleID:  fmt.Sprintf("%d", mr.Rule().ID()),
			Message: mr.Message(),
			Tags:    mr.Rule().Tags(),
			//MatchedData: mr.MatchedDatas(),
		})
	}
	entry.WAFProcessing.MatchedRules = matchedRules

	// Directly set AI variables in the log entry
	entry.WAFProcessing.AIScore = aiScore
	entry.WAFProcessing.AIVerdict = aiVerdict

	jsonData, err := json.Marshal(entry)
	if err != nil {
		c.logger.Printf("Failed to marshal log data: %v", err)
		return
	}
	c.logger.Println(string(jsonData))
}

// LogTransaction now writes in CSV format
func (c *CorazaLogger) LogTransaction(
	tx types.Transaction,
	req *http.Request,
	res *http.Response,
	aiScore float64,    // Directly pass AI score
	aiVerdict string, // Directly pass AI verdict
) {
	clientIP, clientPortStr, _ := net.SplitHostPort(req.RemoteAddr)
	clientPort, _ := strconv.Atoi(clientPortStr)

	serverHost := req.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		serverHost += ":80" // Default to port 80 if not specified
	}
	serverIP, serverPortStr, _ := net.SplitHostPort(serverHost)
	serverPort, _ := strconv.Atoi(serverPortStr)

	// --- Request Body Handling ---
	var reqBodyBytes []byte
	if req.Body != nil {
		var err error
		// Read the entire request body.
		reqBodyBytes, err = ioutil.ReadAll(req.Body)
		if err != nil {
			//log.Printf("Failed to read request body for logging: %v", err)
			// Don't return, still log other data if body fails
		}
		// Restore the body for subsequent handlers
		req.Body = ioutil.NopCloser(bytes.NewBuffer(reqBodyBytes))
	}

	// --- Response Body Handling (placeholder, needs to be passed or retrieved from context) ---
	// As per previous discussion, resBodyBytes is collected in ModifyResponse.
	// For this `LogTransaction` to get it, it would need to be passed as a parameter
	// or stored in context and retrieved here.
	var resBodyBytes []byte // This will be empty if not explicitly retrieved or passed


	// --- Interruption Details ---
	interrupted := tx.IsInterrupted()
	var interruptionRuleID, interruptionAction string
	var interruptionStatus int
	if it := tx.Interruption(); it != nil {
		interruptionRuleID = fmt.Sprintf("%d", it.RuleID)
		interruptionStatus = it.Status
		interruptionAction = it.Action
	}

	// --- Matched Rules Details ---
	var matchedRuleIDs, matchedRuleMessages, matchedRuleTags []string
	for _, mr := range tx.MatchedRules() {
		matchedRuleIDs = append(matchedRuleIDs, fmt.Sprintf("%d", mr.Rule().ID()))
		matchedRuleMessages = append(matchedRuleMessages, mr.Message())
		matchedRuleTags = append(matchedRuleTags, strings.Join(mr.Rule().Tags(), ";")) // Join multiple tags with a semicolon
	}

	// Prepare data for the CSV row
	record := []string{
		time.Now().Format(time.RFC3339),       // Timestamp
		tx.ID(),                               // TransactionID
		clientIP,                              // ClientIP
		strconv.Itoa(clientPort),              // ClientPort
		serverIP,                              // ServerIP
		strconv.Itoa(serverPort),              // ServerPort
		req.Method,                            // RequestMethod
		req.URL.String(),                      // RequestURI
		req.Proto,                             // RequestProtocol
		flattenHeaders(req.Header),            // RequestHeaders
		string(reqBodyBytes),                  // RequestBody
	}

    // --- Defensive Check for Response Data ---
    // Only attempt to access res.StatusCode, res.Proto, res.Header if res is not nil
    if res != nil {
        record = append(record, strconv.Itoa(res.StatusCode)) // ResponseStatus
        record = append(record, res.Proto)                     // ResponseProtocol
        record = append(record, flattenHeaders(res.Header))    // ResponseHeaders
        record = append(record, string(resBodyBytes))          // ResponseBody
    } else {
        // Append empty or placeholder values if no response is available
        record = append(record, "", "", "", "") // For ResponseStatus, Protocol, Headers, Body
    }

	// Append WAF and AI data
	record = append(record,
		strconv.FormatBool(interrupted),      // WAFInterrupted
		interruptionRuleID,                   // InterruptionRuleID
		strconv.Itoa(interruptionStatus),     // InterruptionStatus
		interruptionAction,                   // InterruptionAction
		strconv.Itoa(len(tx.MatchedRules())), // MatchedRulesCount
		strings.Join(matchedRuleIDs, ";"),    // MatchedRulesIDs
		strings.Join(matchedRuleMessages, ";"), // MatchedRulesMessages
		strings.Join(matchedRuleTags, ";"),     // MatchedRulesTags
		fmt.Sprintf("%.2f", aiScore),         // AIScore
		aiVerdict,                            // AIVerdict
	)


	// Write the record to CSV
	if err := c.csvWriter.Write(record); err != nil {
		c.logger.Printf("Failed to write CSV record: %v", err)
	}
	c.csvWriter.Flush() // Ensure data is written to the file
	if err := c.csvWriter.Error(); err != nil {
		c.logger.Printf("CSV writer error: %v", err)
	}
}

// Helper to flatten headers into a single string for CSV
func flattenHeaders(headers http.Header) string {
	var sb strings.Builder
	first := true
	for name, values := range headers {
		if !first {
			sb.WriteString("; ") // Use semicolon to separate headers
		}
		sb.WriteString(name)
		sb.WriteString(":")
		sb.WriteString(strings.Join(values, ",")) // Join multiple values for one header with comma
		first = false
	}
	return sb.String()
}

func (c *CorazaLogger) Close() error {
	c.csvWriter.Flush() // Flush any pending writes before closing
	if err := c.csvWriter.Error(); err != nil {
		c.logger.Printf("Error flushing CSV writer before closing: %v", err)
	}
	return c.file.Close()
}
