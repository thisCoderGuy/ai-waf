import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	// Coraza v3, a ModSecurity-compatible WAF engine.
	"github.com/corazawaf/coraza/v3"
	"github.com/corazawaf/coraza/v3/secreules"
	"github.com/corazawaf/coraza/v3/types"
)


const (
	// AI_MICROSERVICE_URL environment variable will define the AI service URL
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	// TARGET_APP_URL environment variable will define the target web application URL
	targetAppURL = "http://juice-shop:3000"
)


var waf coraza.WAF
var reverseProxy *httputil.ReverseProxy


func init() {
	// Load Coraza WAF rules
	cfg := coraza.NewWAFConfig().
		WithRootFS(secreules.FS).
		WithRequestBodyAccess(true).
		WithResponseBodyAccess(true).
		WithDebugLogger(coraza.DefaultDebugLogger()) // Enable debug logging for Coraza


	var err error
	waf, err = coraza.NewWAF(cfg)
	if err != nil {
		log.Fatalf("Failed to create WAF: %v", err)
	}


	// Load OWASP Core Rule Set (CRS)
	// You might need to adjust the path depending on how secreules.FS is mounted/used.
	// For simplicity, we assume CRS is available at the root of secreules.FS.
	// In a real scenario, you'd configure a specific Coraza.conf to load these.
	log.Println("Loading OWASP CRS...")
	if _, err := waf.NewTransaction().ExecuteMacro(`
		Include @crs-setup.conf.example
		Include @owasp_crs/rules/*.conf
	`); err != nil {
		log.Fatalf("Failed to load CRS: %v", err)
	}
	log.Println("OWASP CRS loaded.")


	// Load our custom Coraza configuration for this proxy
	corazaConf, err := os.ReadFile("/etc/coraza/coraza.conf")
	if err != nil {
		log.Printf("Failed to read custom Coraza config: %v. Continuing without custom config.", err)
	} else {
		if _, err := waf.NewTransaction().ExecuteMacro(string(corazaConf)); err != nil {
			log.Fatalf("Failed to load custom Coraza config: %v", err)
		}
		log.Println("Custom Coraza config loaded.")
	}


	// Setup reverse proxy
	target, err := url.Parse(targetAppURL)
	if err != nil {
		log.Fatalf("Failed to parse target URL: %v", err)
	}
	reverseProxy = httputil.NewSingleHostReverseProxy(target)


	// Custom Director to modify requests before forwarding
	reverseProxy.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.Host = target.Host // Important for target server
		// Remove X-Real-Ip which might be added by previous proxies and can cause issues
		req.Header.Del("X-Real-Ip")
		// Log original remote address
		log.Printf("Forwarding request from %s to %s%s", req.RemoteAddr, req.URL.Host, req.URL.Path)
	}


	reverseProxy.ModifyResponse = func(res *http.Response) error {
		tx := res.Request.Context().Value("coraza_tx").(types.Transaction)
		if tx == nil {
			log.Println("Coraza transaction not found in response context.")
			return nil
		}


		if res.Body != nil {
			bodyBytes, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return fmt.Errorf("failed to read response body: %w", err)
			}
			res.Body.Close() // Close the original body
			res.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes)) // Restore the body for the client


			tx.ProcessResponseBody(bodyBytes)
		}


		if it := tx.Interruption(); it != nil {
			log.Printf("Response Body Coraza Interruption: %s, Rule ID: %d", it.RuleID, it.Status)
			// Note: Blocking on response body is tricky as response is already sent.
			// You might log, alert, or block subsequent requests from the same source.
			// For simplicity, we just log here.
		}


		tx.ProcessLogging()
		return nil
	}


	// Handle errors during reverse proxy operation
	reverseProxy.ErrorHandler = func(rw http.ResponseWriter, req *http.Request, err error) {
		log.Printf("Reverse proxy error: %v", err)
		http.Error(rw, "Bad Gateway", http.StatusBadGateway)
	}
}


func main() {
	http.HandleFunc("/", wafHandler)
	port := ":8080"
	log.Printf("Coraza Proxy listening on port %s, forwarding to %s", port, targetAppURL)
	log.Fatal(http.ListenAndServe(port, nil))
}


func wafHandler(w http.ResponseWriter, r *http.Request) {
	tx := waf.NewTransaction()
	defer tx.Close()


	// Store transaction in context for response processing
	r = r.WithContext(
		// This uses a custom type to avoid context collisions.
		// For simplicity, we use string here, but in production, use context.WithValue(contextKey, value)
		// with a custom type contextKey.
		// For this lab, `(http.HandlerFunc).ServeHTTP` does not pass context down to `reverseProxy.ModifyResponse`.
		// We'll pass the tx directly to reverseProxy.ServeHTTP and handle it there for now.
		// For a more robust solution, use middleware that handles context propagation.
		// For demonstration, we'll make tx globally accessible or pass it differently.
		// A common pattern is to make `reverseProxy` a struct method that holds `tx`
	)


	// Log Coraza transaction ID
	log.Printf("Coraza Transaction ID: %s for %s %s", tx.ID(), r.Method, r.URL.Path)


	// Process request headers
	for k, v := range r.Header {
		tx.AddRequestHeader(k, strings.Join(v, ","))
	}


	// Process request URI and args
	tx.ProcessRequestURL(r.URL.String(), r.Method)


	// Process request body if present
	var bodyBytes []byte
	if r.Body != nil {
		var err error
		bodyBytes, err = ioutil.ReadAll(r.Body)
		if err != nil {
			log.Printf("Failed to read request body: %v", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}
		// Restore the body for the next handler (AI service and reverse proxy)
		r.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
	}
	tx.ProcessRequestBody(bodyBytes)


	// --- AI Microservice Call ---
	// This is the real-time AI integration point
	aiVerdict := "benign" // Default to benign
	if len(bodyBytes) > 0 || r.URL.RawQuery != "" || r.Method == http.MethodPost || r.Method == http.MethodPut {
		// Only send relevant requests to AI for efficiency
		aiReq := map[string]string{
			"method": r.Method,
			"path":   r.URL.Path,
			"query":  r.URL.RawQuery,
			"body":   string(bodyBytes),
			// Add more fields as needed: headers, client IP, etc.
		}
		aiReqJSON, _ := json.Marshal(aiReq)


		log.Printf("Sending request to AI microservice: %s", aiMicroserviceURL)
		aiResp, err := http.Post(aiMicroserviceURL, "application/json", bytes.NewBuffer(aiReqJSON))
		if err != nil {
			log.Printf("Error calling AI microservice: %v", err)
		} else {
			defer aiResp.Body.Close()
			if aiResp.StatusCode == http.StatusOK {
				aiRespBody, _ := ioutil.ReadAll(aiResp.Body)
				var aiRespMap map[string]interface{}
				if err := json.Unmarshal(aiRespBody, &aiRespMap); err == nil {
					if verdict, ok := aiRespMap["verdict"].(string); ok {
						aiVerdict = verdict
						log.Printf("AI Microservice verdict: %s", aiVerdict)
					}
					if score, ok := aiRespMap["score"].(float64); ok {
						log.Printf("AI Microservice score: %.2f", score)
						// Example: Inject AI score into Coraza transaction
						// This allows Coraza rules to use the AI's score.
						// You would need a custom Coraza rule like:
						// SecRule TX:AI_SCORE "@gt 0.8" "id:1000,phase:2,deny,msg:'AI detected high score'"
						tx.Collection(types.TxCollection).Set("AI_SCORE", fmt.Sprintf("%.2f", score))
					}
				}
			} else {
				log.Printf("AI Microservice returned non-OK status: %d", aiResp.StatusCode)
			}
		}
	}


	// --- Coraza Interruption ---
	// Process the transaction and check for interruption
	it := tx.ProcessRequest()
	if it == nil && aiVerdict == "malicious" {
		// If Coraza didn't block it, but AI says it's malicious, we can force a block.
		// Or, inject a custom rule based on AI verdict for Coraza to process.
		log.Println("AI detected malicious request, forcing Coraza interruption.")
		tx.Interruption("AI_DECISION", types.ActionDisruptiveStatus(403)) // Force a 403
	}


	// Re-check for interruption after AI decision (if AI forces one)
	it = tx.Interruption()
	if it != nil {
		log.Printf("Coraza Interruption: %s, Rule ID: %d, Status: %d", it.RuleID, it.RuleID, it.Status)
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(it.Status)
		_, _ = w.Write([]byte(fmt.Sprintf("Coraza WAF Blocked: %s (Rule ID: %d)", it.RuleID, it.RuleID)))
		tx.ProcessLogging()
		return
	}


	// If no interruption, forward the request to the target application
	reverseProxy.ServeHTTP(w, r)


	// Note: Response body inspection is handled in reverseProxy.ModifyResponse
	// The tx.ProcessLogging() for the full transaction is called there too.
}
