package main

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"

	"github.com/corazawaf/coraza/v3/types"
)

// CallAIMicroservice makes a request to the AI microservice for threat classification
func CallAIMicroservice(r *http.Request, bodyBytes []byte, tx types.Transaction) (string, float64) {
	aiVerdict := "Not Evaluated" // Default verdict
	aiScore := -1.0        // Default score

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
			//log.Printf("Sending request to AI microservice: %s", GetAIMicroserviceURL())
			// Make an HTTP POST request to the AI microservice.
			aiResp, err := http.Post(GetAIMicroserviceURL(), "application/json", bytes.NewBuffer(aiReqJSON))
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
								//log.Printf("AI Microservice verdict: %s", aiVerdict)
							}
							if score, ok := aiRespMap["score"].(float64); ok {
								aiScore = score
								//log.Printf("AI Microservice score: %.2f", aiScore)
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

	return aiVerdict, aiScore
}