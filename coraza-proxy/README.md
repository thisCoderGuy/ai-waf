# Generating Datasets

## Traffic Generation Timing

In `./kali/startup.sh`
- Change delay before traffic is generated: `DELAY_SECONDS=60`
- Change duration of traffic generation (x2):  `--run-time 10m`  

## Type of Traffic Generated

In `./kali/locust_tests/config.py`

- Change the desired type of traffic Generated
   
   `USER_TASK_WEIGHTS = { # ints only

        "LegitimateUser": 12,

        "SQLiAttacker":0,

        "XSSAttacker": 0,

        "DirectoryTraversalAttacker": 0,

        "EnumerationAttacker": 0,

        "CSRFAttacker": 0,

    }`

## Log Format

In ./coraza-proxy/config.go

- Change the **log file format** `loggerFormat` (”csv” or ”json”)
- Change the **log file name** `loggerPath` (e.g. “/var/log/coraza/coraza-audit-benign.csv”)
- Change the **labels** `DefaultAIVerdictLabel` and `DefaultAIVulnerabilityTypeLabel` for the type of datasets being generated
  - `DefaultAIVerdictLabel (E.g. "benign", malicious)
  - DefaultAIVulnerabilityTypeLabel (E.g. “none”, sqli, xss, etc)

`const loggerFormat = "csv"

const loggerPath ="/var/log/coraza/coraza-audit-benign.csv"

const (

	DefaultAIVerdictLabel         = "benign" // benign or malicious

	DefaultAIVulnerabilityTypeLabel = "none"  //none, sqli, xss, etc.

)`


