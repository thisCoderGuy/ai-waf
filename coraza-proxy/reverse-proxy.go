package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/corazawaf/coraza/v3/types"
)

var reverseProxy *httputil.ReverseProxy

func InitializeReverseProxy() error {
	// Setup Reverse Proxy that will be used by the wafhandler
	// Parse the target application URL.
	target, err := url.Parse(GetTargetAppURL())
	if err != nil {
		return fmt.Errorf("failed to parse target URL: %w", err)
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
		//log.Printf(">>>>> Forwarding request from %s to %s%s", req.RemoteAddr, req.URL.Host, req.URL.Path)
	}

	// ModifyResponse function processes the response from the target application.
	reverseProxy.ModifyResponse = func(res *http.Response) error {
		//GetGlobalLogger().LogInfo("In ReverseProxy")

		req := res.Request // Get the original request from the response

		// Retrieve the Coraza transaction from the request context.
		tx, ok := req.Context().Value(corazaTxContextKey).(types.Transaction)
		if !ok || tx == nil {
			log.Println("Coraza transaction not found in response context or invalid type. Cannot process response body.")
			return nil
		}

		
		var aiScoreFromContext float64 = 0.0 
		var aiVerdictFromContext string = "not evaluated" 

		if val := req.Context().Value(aiScoreContextKey); val != nil {
			if score, ok := val.(float64); ok { // Type assertion to the stored type
				aiScoreFromContext = score
			}
		}
		if val := req.Context().Value(aiVerdictContextKey); val != nil {
			if verdict, ok := val.(string); ok { // Type assertion to the stored type
				aiVerdictFromContext = verdict
			}
		}

		// Retrieve request body bytes from context if stored earlier by wafHandler
		/*
		var reqBodyBytes []byte
		if val := req.Context().Value(reqBodyContextKey); val != nil {
			if b, ok := val.([]byte); ok {
				reqBodyBytes = b
			}
		}
			*/

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
			//log.Printf("Response Headers Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)
			// Modify the response being sent back to the client based on the interruption.
			res.StatusCode = it.Status // Set the response status code based on Coraza's interruption
			// Override the response body with a WAF blocked message.
			blockedMsg := fmt.Sprintf("Coraza WAF Blocked Response Headers: %s (Rule ID: %s)", it.Action, it.RuleID)
			res.Body = ioutil.NopCloser(bytes.NewBufferString(blockedMsg))
			res.ContentLength = int64(len(blockedMsg))   // Update content length for the new body
			res.Header.Set("Content-Type", "text/plain") // Set content type for clarity
			res.Header.Del("Content-Length")             // Remove original Content-Length, it will be recalculated by Go's HTTP server
			res.Header.Del("Content-Encoding")           // Clear any encoding from original response
		
			// Handle the interruption, if any
			if it.Action == "deny" {
				tx.ProcessLogging()
				if logger := GetGlobalLogger(); logger != nil {
					logger.LogTransaction(tx, req, res, aiScoreFromContext, aiVerdictFromContext)
				}
				return nil
			}
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
			//log.Printf("Response Body Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)
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

			// Handle the interruption, if any
			if it.Action == "deny" {
				tx.ProcessLogging()
				if logger := GetGlobalLogger(); logger != nil {
					logger.LogTransaction(tx, req, res, aiScoreFromContext, aiVerdictFromContext)
				}
				return nil
			}
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Phase 5. Logging
		// This is the final logging phase, executed regardless of interruptions.
		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Process logging for the entire transaction (request and response phases).
		tx.ProcessLogging()
		//log.Println(">>>>> Response for request  was transmitted!")
		return nil
	}

	// ErrorHandler function handles errors that occur during the reverse proxy operation.
	reverseProxy.ErrorHandler = func(rw http.ResponseWriter, req *http.Request, err error) {
		log.Printf("Reverse proxy error for request %s %s: %v", req.Method, req.URL.Path, err)
		http.Error(rw, "Bad Gateway", http.StatusBadGateway)
	}

	return nil
}

func GetReverseProxy() *httputil.ReverseProxy {
	return reverseProxy
}