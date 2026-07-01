#!/usr/bin/env python3
"""Wait for milestones.json and create submission document"""

import json
import time
import os
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
milestone_file = script_dir / "crawled_data" / "milestones.json"
output_dir = script_dir

print("Waiting for crawler to complete and milestones.json to be created...")

# Wait for milestones.json (timeout after 2 hours)
timeout = 7200  # 2 hours
start_time = time.time()
while not milestone_file.exists():
    elapsed = time.time() - start_time
    if elapsed > timeout:
        print(f"Timeout: milestones.json not created after {timeout}s")
        exit(1)
    print(f"  [{int(elapsed)}s] Waiting...", end="\r")
    time.sleep(5)

print("\n✓ milestones.json created!")

# Read milestones
with open(milestone_file, 'r') as f:
    data = json.load(f)

milestones = data.get("milestones", {})
total_success = data.get("total_success", 0)
total_attempted = data.get("total_attempted", 0)

print(f"\nMilestones reached:")
print(f"  Page 333: {milestones.get('333', 'N/A')} documents downloaded")
print(f"  Page 666: {milestones.get('666', 'N/A')} documents downloaded")
print(f"  Page 1000: {milestones.get('1000', 'N/A')} documents downloaded")
print(f"\nTotal: {total_success}/{total_attempted} successful")

# Create submission document
submission_doc = output_dir / "Submission.md"
with open(submission_doc, 'w') as f:
    f.write("""# Fermilab Web Crawler - Submission Report

## Project Overview
Extended web crawler that downloads up to 1,000 public Fermilab web pages from multiple domains including:
- www.fnal.gov
- news.fnal.gov
- education.fnal.gov
- history.fnal.gov
- And others

## Key Statistics

### Documents Downloaded at Milestones

1. **At Page 333 Checkpoint**: {} documents successfully downloaded
2. **At Page 666 Checkpoint**: {} documents successfully downloaded
3. **At Page 1000 (Completion)**: {} documents successfully downloaded

### Overall Results
- **Total Pages Attempted**: {}
- **Total Pages Downloaded Successfully**: {}
- **Success Rate**: {:.1f}%

## Output Files (located in `crawled_data/` directory)

### Data Files
- `download_report_1000.csv` - CSV report with metadata for all 1,000 pages
- `milestones.json` - JSON file with milestone tracking data
- `progress.log` - Timestamped progress log

### Directories
- `html_pages/` - Downloaded HTML files ({} files)
- `text_pages/` - Extracted text from HTML ({} files)

## Methodology

The crawler:
1. Starts from 10 seed URLs across different Fermilab domains
2. Extracts links from each page and follows internal links only
3. Implements polite delays (1.5 seconds) between requests
4. Saves both raw HTML and extracted text for each page
5. Tracks progress with timestamped logs and milestone markers
6. Records HTTP status codes, page titles, and file sizes
7. Continues until 1,000 pages are downloaded or link queue is exhausted

## Implementation
- **Language**: Python 3
- **Libraries**: requests, BeautifulSoup4
- **Features**: 
  - Breadth-first search link discovery
  - Domain-restricted crawling
  - Error handling and retry logic
  - CSV reporting
  - Timestamped milestone tracking
""".format(
    milestones.get('333', 'N/A'),
    milestones.get('666', 'N/A'),
    milestones.get('1000', 'N/A'),
    total_attempted,
    total_success,
    (total_success / total_attempted * 100) if total_attempted > 0 else 0,
    len(list((script_dir / "crawled_data" / "html_pages").glob("*.html"))),
    len(list((script_dir / "crawled_data" / "text_pages").glob("*.txt")))
))

print(f"\n✓ Submission document created: {submission_doc}")
