package main

import (
	"bytes"
	"context" // Import context package for proper request context handling
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	//"sync"
	"embed"
	"net"
	"strconv"

	"github.com/corazawaf/coraza/v3"
	"github.com/corazawaf/coraza/v3/types"
)

// --- Custom Log Entry Structures ---
// These structs define the schema for your JSON log entries.

type CorazaLogEntry struct {
	Timestamp     time.Time            `json:"timestamp"`
	TransactionID string               `json:"transaction_id"`
	ClientIP      string               `json:"client_ip"`
	ClientPort    int                  `json:"client_port"`
	ServerIP      string               `json:"server_ip"`
	ServerPort    int                  `json:"server_port"`
	Request       RequestLogData       `json:"request"`
	Response      ResponseLogData      `json:"response"`
	WAFProcessing WAFProcessingLogData `json:"waf_processing"`
}

type RequestLogData struct {
	Method   string              `json:"method"`
	URI      string              `json:"uri"`
	Protocol string              `json:"protocol"`
	Headers  map[string][]string `json:"headers"`
	Body     string              `json:"body,omitempty"` // Use omitempty for optional body
}

type ResponseLogData struct {
	StatusCode int                 `json:"status_code"`
	Protocol   string              `json:"protocol"`
	Headers    map[string][]string `json:"headers"`
	Body       string              `json:"body,omitempty"`
}

type WAFProcessingLogData struct {
	Interrupted         bool                 `json:"interrupted"`
	InterruptionDetails *InterruptionLogData `json:"interruption_details,omitempty"`
	MatchedRules        []MatchedRuleLogData `json:"matched_rules"`
	AIScore             float64              `json:"ai_score,omitempty"`
	AIVerdict           string               `json:"ai_verdict,omitempty"`
	CorazaDebugLog      string               `json:"coraza_debug_log,omitempty"` // Optional: if you capture debug logs
}

type InterruptionLogData struct {
	RuleID  string `json:"rule_id"`
	Status  int    `json:"status"`
	Action  string `json:"action"`
	Message string `json:"message,omitempty"`
}

type MatchedRuleLogData struct {
	RuleID      string   `json:"rule_id"`
	Severity    int      `json:"severity"`
	Message     string   `json:"message,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	MatchedData string   `json:"matched_data,omitempty"`
}

// --- CorazaLogger Struct and Methods ---
type CorazaLogger struct {
	file   *os.File
	logger *log.Logger
}

func NewCorazaLogger(filename string) (*CorazaLogger, error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}

	return &CorazaLogger{
		file:   file,
		logger: log.New(file, "", log.LstdFlags),
	}, nil
}

func (c *CorazaLogger) LogTransaction(
	tx types.Transaction,
	req *http.Request,
	reqBody []byte,
	res *http.Response,
	resBody []byte,
	aiScore float64, // Directly pass AI score
	aiVerdict string, // Directly pass AI verdict
	clientIP string, // Directly pass client IP
	clientPort int, // Directly pass client port
	serverIP string, // Directly pass server IP
	serverPort int, // Directly pass server port
) {
	entry := CorazaLogEntry{
		Timestamp:     time.Now(),
		TransactionID: tx.ID(),
		ClientIP:      clientIP,
		ClientPort:    clientPort,
		ServerIP:      serverIP,
		ServerPort:    serverPort,
	}

	// Request Data
	entry.Request = RequestLogData{
		Method:   req.Method,
		URI:      req.URL.String(),
		Protocol: req.Proto,
		Headers:  req.Header, // Raw headers
		Body:     string(reqBody),
	}

	// Response Data
	if res != nil {
		entry.Response = ResponseLogData{
			StatusCode: res.StatusCode,
			Protocol:   res.Proto,
			Headers:    res.Header, // Raw headers
			Body:       string(resBody),
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

func (c *CorazaLogger) Close() error {
	return c.file.Close()
}

// Define a custom type for context keys to avoid collisions.
// This is a best practice for passing values through http.Request contexts.
// corazaTxContextKey is used to store and retrieve the Coraza transaction from the context.
type contextKey string

const corazaTxContextKey contextKey = "coraza-transaction"

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)

var waf coraza.WAF
var reverseProxy *httputil.ReverseProxy
var corazaLogger CorazaLogger

// This annotation below embeds the CRS rules in the owasp-crs-v4/ folder
//
//go:embed owasp-crs-v4/*
var crs embed.FS

func GetEmbeddedCRSFS() embed.FS {
	return crs
}

func init() {

	customConfigPath := "/etc/coraza/coraza.conf"
	customDirectives, err := loadCustomDirectives(customConfigPath)
	if err != nil {
		log.Fatalf("Failed to load custom Coraza configuration: %v", err)
	}

	// Load OWASP CRS rules using Coraza's `Include` directive
	// The paths are relative to the root of the embedded filesystem provided to WithRootFS.
	// In our case, the root is 'owasp-crs-v4' itself.
	crsLoadScript := `
		SecRuleEngine On
		Include owasp-crs-v4/crs-setup.conf.example
		Include owasp-crs-v4/rules/*.conf
	`

	cfg := coraza.NewWAFConfig().
		WithRequestBodyAccess().  // Enable access to the request body for WAF inspection
		WithResponseBodyAccess(). // Enable access to the response body for WAF inspection
		//WithDebugLogger(myDebugLogger). // Enable debug logging for Coraza
		WithRootFS(GetEmbeddedCRSFS()). // Call the function from rules.go
		WithDirectives(crsLoadScript + "\n" + customDirectives)

	waf, err = coraza.NewWAF(cfg)
	if err != nil {
		log.Fatalf("Failed to create WAF: %v", err)
	}

	fmt.Println("Coraza WAF initialized successfully with OWASP CRS v4 embedded rules!")

	// Setup Reverse Proxy that will be used by the wafhandler
	// Parse the target application URL.
	target, err := url.Parse(targetAppURL)
	if err != nil {
		log.Fatalf("Failed to parse target URL: %v", err)
	}
	// Create a new single host reverse proxy that forwards requests to the target.
	reverseProxy = httputil.NewSingleHostReverseProxy(target)

	// Custom Director function to modify requests before forwarding them to the target.
	reverseProxy.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.Host = target.Host // Important for the target server to receive the correct Host header
		// Remove X-Real-Ip which might be added by previous proxies and can cause issues.
		req.Header.Del("X-Real-Ip")
		// Log the original remote address and the forwarding destination.
		log.Printf("Forwarding request from %s to %s%s", req.RemoteAddr, req.URL.Host, req.URL.Path)
	}

	// ModifyResponse function processes the response from the target application.
	reverseProxy.ModifyResponse = func(res *http.Response) error {
		// Retrieve the Coraza transaction from the request context.
		tx, ok := res.Request.Context().Value(corazaTxContextKey).(types.Transaction)
		if !ok || tx == nil {
			log.Println("Coraza transaction not found in response context or invalid type. Cannot process response body.")
			return nil
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Phase 3. ProcessResponseHeaders() examines the HTTP response headers sent by the backend server before the response body is sent.
		// Coraza evaluates these headers against configured WAF rules for phase 3.
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Iterate through response headers and add them to Coraza transaction.
		for k, v := range res.Header {
			tx.AddResponseHeader(k, strings.Join(v, ","))
		}

		it := tx.ProcessResponseHeaders(res.StatusCode, res.Proto)

		// If an interruption occurs in Phase 3 (Response Headers), handle it.
		if it != nil {
			log.Printf("Response Headers Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)
			// Modify the response being sent back to the client based on the interruption.
			res.StatusCode = it.Status // Set the response status code based on Coraza's interruption
			// Override the response body with a WAF blocked message.
			blockedMsg := fmt.Sprintf("Coraza WAF Blocked Response Headers: %s (Rule ID: %s)", it.Action, it.RuleID)
			res.Body = ioutil.NopCloser(bytes.NewBufferString(blockedMsg))
			res.ContentLength = int64(len(blockedMsg))   // Update content length for the new body
			res.Header.Set("Content-Type", "text/plain") // Set content type for clarity
			res.Header.Del("Content-Length")             // Remove original Content-Length, it will be recalculated by Go's HTTP server
			res.Header.Del("Content-Encoding")           // Clear any encoding from original response
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Phase 4. ProcessResponseBody() inspects the body content of the HTTP response from the backend server.
		// Applies WAF rules configured for phase 4 (Response Body phase).
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Read and process the response body if it exists.
		if res.Body != nil {
			bodyBytes, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return fmt.Errorf("failed to read response body: %w", err)
			}
			res.Body.Close() // Close the original body to prevent resource leaks
			// Restore the body for the client by creating a new io.ReadCloser from the bytes.
			res.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))

			// Process the response body with Coraza.
			tx.ProcessResponseBody()
		}

		// Check for any interruptions triggered by response body inspection.
		if it := tx.Interruption(); it != nil {
			log.Printf("Response Body Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)
			// Note: Blocking on response body is tricky as the response might have already started
			// being sent to the client. In such cases, you might log, alert, or block
			// subsequent requests from the same source, rather than attempting to block this one.
			// For simplicity, we just log here.
			// Modify the response being sent back to the client.
			res.StatusCode = it.Status
			blockedMsg := fmt.Sprintf("Coraza WAF Blocked Response Body: %s (Rule ID: %s)", it.Action, it.RuleID)
			res.Body = ioutil.NopCloser(bytes.NewBufferString(blockedMsg))
			res.ContentLength = int64(len(blockedMsg))
			res.Header.Set("Content-Type", "text/plain")
			res.Header.Del("Content-Length")
			res.Header.Del("Content-Encoding")
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Phase 5. Logging
		// This is the final logging phase, executed regardless of interruptions.
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Process logging for the entire transaction (request and response phases).
		tx.ProcessLogging()
		return nil
	}

	// ErrorHandler function handles errors that occur during the reverse proxy operation.
	reverseProxy.ErrorHandler = func(rw http.ResponseWriter, req *http.Request, err error) {
		log.Printf("Reverse proxy error for request %s %s: %v", req.Method, req.URL.Path, err)
		http.Error(rw, "Bad Gateway", http.StatusBadGateway)
	}

}

func loadCustomDirectives(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	content, err := ioutil.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("failed to read config file: %w", err)
	}

	return string(content), nil
}

func main() {

	// Create logger that saves to file
	corazaLogger, err := NewCorazaLogger("/var/log/coraza/coraza-audit.log")
	if err != nil {
		log.Fatal(err)
	}
	defer corazaLogger.Close()

	// Register the wafHandler to handle all incoming HTTP requests.
	http.HandleFunc("/", wafHandler) //  "/": This is the URL path being registered. In this case, it's the root path (/), meaning this handler will be invoked for all incoming requests to the server regardless of the path.
	port := ":8080"
	log.Printf("Coraza Proxy listening on port %s, forwarding to %s", port, targetAppURL)
	// Start the HTTP server. This call blocks until the server stops or an error occurs.
	log.Fatal(http.ListenAndServe(port, nil))

}

// wafHandler processes incoming HTTP requests through Coraza WAF and an AI microservice,
// then forwards them to the target application via a reverse proxy, if not intercepted.

// Client Request
//
//	↓
//
// [wafHandler]
//
//		↳ Coraza WAF: analyze  request headers, request body, URI (Phases 1 & 2)
//		↳ AI Microservice: classify threat level
//		↳ If malicious → block, enrich context with ai_malicious code, then  log (to be sent to wazuh)
//		↳ If safe → pass to reverseProxy
//	    [reverseProxy]
//		    ↳ Forward to backend (e.g., Juice Shop)
//		    ↳ Get response
//		    ↳ Coraza WAF: analyze response (Phase 3 & 4)
//		    ↳ Return response to client, if not intercepted.
func wafHandler(w http.ResponseWriter, r *http.Request) {
	// Create a new Coraza transaction for each incoming request.
	tx := waf.NewTransaction()
	defer tx.Close() // Ensure the transaction is closed to release resources

	// Store the Coraza transaction in the request context.
	// This allows the transaction to be retrieved later in `reverseProxy.ModifyResponse`.
	ctx := context.WithValue(r.Context(), corazaTxContextKey, tx)
	r = r.WithContext(ctx)

	// Log the Coraza transaction ID for debugging and tracing.
	log.Printf("Coraza Transaction ID: %s for %s %s", tx.ID(), r.Method, r.URL.Path)

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0. Coraza Transaction Initial Setup. Does not trigger any rule.
	// Phase 0.0 ProcessConnection() is used to log metadata about the TCP connection that initiated the request (Client IP and port, Server IP and port).
	// Parse client IP and port
	clientIP, clientPortStr, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		http.Error(w, "Invalid remote address", http.StatusInternalServerError)
		return
	}
	clientPort, _ := strconv.Atoi(clientPortStr)

	// Parse server IP and port
	serverHost := r.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		// Append default port if not specified
		serverHost += ":80"
	}
	serverIP, serverPortStr, err := net.SplitHostPort(serverHost)
	if err != nil {
		http.Error(w, "Invalid host address", http.StatusInternalServerError)
		return
	}
	serverPort, _ := strconv.Atoi(serverPortStr)

	// Now call ProcessConnection
	tx.ProcessConnection(clientIP, clientPort, serverIP, serverPort)
	if it := tx.Interruption(); it != nil {
		log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {

			http.Error(w, "WAF headers check failed", http.StatusBadRequest)
			return
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0.1 ProcessURI() is used to used to initialize the transaction with key elements of the HTTP request line
	// It sets the following: Request URI (e.g., /login?user=admin), HTTP method (e.g., GET, POST), HTTP version (e.g., HTTP/1.1, HTTP/2)
	// Process request URI and arguments with Coraza.
	uri := r.URL.String()
	if r.URL.RawQuery != "" {
		uri = uri + "?" + r.URL.RawQuery
	}
	tx.ProcessURI(uri, r.Method, r.Proto)

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0.2 AddRequestHeader() is used to inject or append HTTP request headers into the transaction object
	// Process request headers with Coraza.
	for k, v := range r.Header {
		tx.AddRequestHeader(k, strings.Join(v, ","))
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 1. ProcessRequestHeaders() evaluates the HTTP request headers against ModSecurity/OWASP rules.
	it := tx.ProcessRequestHeaders()

	// Check for any interruptions triggered by request inspection.
	if it := tx.Interruption(); it != nil {
		log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "WAF headers check failed", http.StatusBadRequest)
			tx.ProcessLogging()
			return
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 2. ProcessRequestBody() inspects the HTTP request body (e.g., POST data, JSON payloads, form submissions)  for malicious content based on the loaded WAF rules.
	// Process request body if present.
	var bodyBytes []byte
	if r.Body != nil {
		var err error
		// Read the entire request body.
		bodyBytes, err = ioutil.ReadAll(r.Body)
		if err != nil {
			log.Printf("Failed to read request body: %v", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return // Exit if body cannot be read
		}
		// Restore the body for subsequent handlers (AI service and reverse proxy)
		// as `ioutil.ReadAll` consumes the original `r.Body`.
		r.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
	}
	tx.ProcessRequestBody()

	// Check for any interruptions triggered by request inspection.
	if it := tx.Interruption(); it != nil {
		log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "Request blocked by WAF", http.StatusForbidden)
			tx.ProcessLogging()
			return
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// New Phase: AI Microservice Call
	// This section integrates with an external AI microservice for additional threat classification.
	aiVerdict := "benign" // Default verdict
	aiScore := 0.0        //Default score
	// Only send requests with body, query parameters, or specific methods to AI for efficiency.
	if len(bodyBytes) > 0 || r.URL.RawQuery != "" || r.Method == http.MethodPost || r.Method == http.MethodPut {
		// Prepare the request payload for the AI microservice.
		aiReq := map[string]string{
			"method": r.Method,
			"path":   r.URL.Path,
			"query":  r.URL.RawQuery,
			"body":   string(bodyBytes),
			// Additional request details (headers, client IP) could be added here.
			//TODO: format data point for prediction!!!!!
		}
		aiReqJSON, err := json.Marshal(aiReq) // Marshal to JSON, ignoring error for simplicity in example
		if err != nil {
			log.Printf("Error marshaling AI request: %v", err)
		} else {
			log.Printf("Sending request to AI microservice: %s", aiMicroserviceURL)
			// Make an HTTP POST request to the AI microservice.
			aiResp, err := http.Post(aiMicroserviceURL, "application/json", bytes.NewBuffer(aiReqJSON))
			if err != nil {
				log.Printf("Error calling AI microservice: %v", err)
			} else {
				defer aiResp.Body.Close() // Ensure response body is closed
				if aiResp.StatusCode == http.StatusOK {
					aiRespBody, err := ioutil.ReadAll(aiResp.Body) // Read AI response
					if err != nil {
						log.Printf("Error reading AI response body: %v", err)
					} else {
						var aiRespMap map[string]interface{}
						// Unmarshal AI response JSON.
						if err := json.Unmarshal(aiRespBody, &aiRespMap); err == nil {
							if verdict, ok := aiRespMap["verdict"].(string); ok {
								aiVerdict = verdict
								tx.AddRequestHeader("aiVerdict", aiVerdict)
								log.Printf("AI Microservice verdict: %s", aiVerdict)
							}
							if score, ok := aiRespMap["score"].(float64); ok {
								aiScore = score

								log.Printf("AI Microservice score: %.2f", aiScore)
								tx.AddRequestHeader("aiScore", strconv.FormatFloat(aiScore, 'f', 2, 64))
							}
						} else {
							log.Printf("Error unmarshaling AI response: %v", err)
						}
					}
				} else {
					log.Printf("AI Microservice returned non-OK status: %d", aiResp.StatusCode)
				}
			}
		}
	}

	// --- Coraza Interruption and AI Decision Enforcement ---
	it = tx.Interruption()

	// If Coraza didn't block the request, but the AI microservice deemed it malicious,
	// we can force an interruption based on the AI's verdict.
	if it == nil && aiVerdict == "malicious" {
		log.Println("AI detected malicious request, forcing Coraza interruption.")
		// Manually create an interruption struct to simulate a WAF block.
		it = &types.Interruption{
			RuleID: 666,    // A custom identifier for blocks initiated by AI
			Status: 403,    // HTTP status code (e.g., Forbidden)
			Action: "deny", // The action taken
		}
	}

	// Check for interruption again (either from Coraza rules or AI-forced).
	// If `it` is still nil, it means no interruption occurred.
	if it != nil {
		log.Printf("Coraza WAF Blocked Request! Action: %s, Rule ID: %d, Status: %d", it.Action, it.RuleID, it.Status)
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(it.Status)
		// Write a response to the client indicating the block.
		_, _ = w.Write([]byte(fmt.Sprintf("Coraza WAF Blocked: %s (Rule ID: %d)", it.Action, it.RuleID)))
		// Phase 5. Logging
		tx.ProcessLogging() // Ensure logging is processed for blocked requests.
		//TODO
		// Log the full transaction details using the custom logger.
		//	corazaLogger.LogTransaction(tx, res.Request, reqBody, res, resBodyBytes, aiScore, aiVerdict, clientIP, clientPort, serverIP, serverPort)

		return // Stop processing and return the response.
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// If no interruption occurred, forward the request to the target application by using the reverse proxy.
	reverseProxy.ServeHTTP(w, r)
	// Note: Phase 3 (Response Headers Inspection), Phase 4 (Response body inspection) and Phase 5 (`tx.ProcessLogging()`) for the full transaction
	// are handled within the `reverseProxy.ModifyResponse` function.
}
