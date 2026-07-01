# Fermilab Web Crawler - Submission Report

## Project Overview
Extended web crawler that downloads up to 1,000 public Fermilab web pages from multiple domains including:
- www.fnal.gov
- news.fnal.gov
- education.fnal.gov
- history.fnal.gov
- And others

## Key Statistics

### Documents Downloaded at Milestones

1. **At Page 333 Checkpoint**: 296 documents successfully downloaded
2. **At Page 666 Checkpoint**: 606 documents successfully downloaded
3. **At Page 1000 (Completion)**: 936 documents successfully downloaded

### Overall Results
- **Total Pages Attempted**: 1000
- **Total Pages Downloaded Successfully**: 936
- **Success Rate**: 93.6%

## Output Files (located in `crawled_data/` directory)

### Data Files
- `download_report_1000.csv` - CSV report with metadata for all 1,000 pages
- `milestones.json` - JSON file with milestone tracking data
- `progress.log` - Timestamped progress log

### Directories
- `html_pages/` - Downloaded HTML files (2233 files)
- `text_pages/` - Extracted text from HTML (2233 files)

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
