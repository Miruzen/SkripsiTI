#!/usr/bin/env python3
"""
dailyfxscraper.py
Requests/BeautifulSoup-based scraper for DailyForex EUR/USD article pages (pages 247–296).
Now auto-fixes missing numeric article IDs and extracts directly from JSON-LD via HTTP requests.
Saves: author, date, title, url, content → output/dailyforex_articles.xlsx
"""

import os
import shutil
import time
import json
import random
import re
from datetime import datetime

# Import non-browser based libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# ---------------- CONFIG ----------------
START_PAGE = 3
END_PAGE = 35

LISTING_TEMPLATE = "https://www.dailyforex.com/articles/currency-pairs/english/5754/{}"
BASE_DOMAIN = "https://www.dailyforex.com"

OUTPUT_XLSX = "output/dailyforex_articles_uji.xlsx"
BACKUP_DIR = "output/backups"

SAVE_INTERVAL_PAGES = 1 	 # auto-save every N listing pages processed
BACKUP_EVERY_PAGES = 10 	 # create backup every N pages processed (timestamped copy)
DELAY_LISTING = (3.0, 5.0) 	# seconds delay before fetching each listing page
DELAY_ARTICLE = (5.0, 8.0) 	# seconds delay between article visits

# DEBUG_MODE is no longer needed since we are not launching a visible browser.

USER_AGENTS = [
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]
# ----------------------------------------

os.makedirs("output", exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


# =====================================================
# Utility Functions
# =====================================================

def load_existing_urls():
	"""Load already scraped URLs to resume without duplicates."""
	if os.path.exists(OUTPUT_XLSX):
		try:
			df = pd.read_excel(OUTPUT_XLSX)
			if "url" in df.columns:
				# Use the canonical URL for tracking
				return set(df["url"].dropna().astype(str).tolist()), df 
			return set(), df
		except Exception as e:
			print(f"⚠️ Warning: could not read existing XLSX ({e})")
			return set(), pd.DataFrame(columns=["author", "date", "title", "url", "content"])
	return set(), pd.DataFrame(columns=["author", "date", "title", "url", "content"])


def backup_output():
	"""Backup Excel file every few pages."""
	if os.path.exists(OUTPUT_XLSX):
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		dst = os.path.join(BACKUP_DIR, f"backup_{stamp}.xlsx")
		try:
			shutil.copy2(OUTPUT_XLSX, dst)
			print(f"🗂️ Backup created → {dst}")
		except Exception as e:
			print(f"⚠️ Backup failed: {e}")


def sanitize_text(x):
	if x is None:
		return ""
	# Replace newlines and carriage returns with spaces, then strip leading/trailing whitespace
	return str(x).replace("\r", " ").replace("\n", " ").strip()


def save_append_excel(existing_df, new_articles):
	"""Append and deduplicate scraped articles."""
	if not new_articles:
		print("ℹ️ Nothing new to save.")
		return

	df_new = pd.DataFrame(new_articles)
	for col in ["author", "date", "title", "url", "content"]:
		if col not in df_new.columns:
			df_new[col] = ""

	df_new = df_new.fillna("").applymap(sanitize_text)
	if existing_df is None or existing_df.empty:
		df_combined = df_new
	else:
		df_combined = pd.concat([existing_df, df_new], ignore_index=True)

	if "url" in df_combined.columns:
		df_combined.drop_duplicates(subset="url", inplace=True)

	df_combined.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
	print(f"💾 Saved {len(df_combined)} articles → {OUTPUT_XLSX}")


# =====================================================
# Article Extraction (Rewritten for requests)
# =====================================================

def get_session_headers():
    """Generates a common set of browser headers for better stealth."""
    user_agent = random.choice(USER_AGENTS)
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def extract_article(url, headers, timeout_sec=60):
	"""Fetch an article via requests and extract info via JSON-LD or fallback DOM using BeautifulSoup."""
	out = {"author": "", "date": "", "title": "", "url": url, "content": ""}

	try:
		# Use stream=True to prevent issues with large responses, then raise_for_status to check for 4xx/5xx
		response = requests.get(url, headers=headers, timeout=timeout_sec, allow_redirects=True)
		response.raise_for_status()

		# Capture the final URL after redirects (this is the canonical URL)
		out["url"] = response.url 
		
		# Parse the content
		soup = BeautifulSoup(response.content, 'lxml')

		# ---------------- Try JSON-LD extraction ----------------
		json_ld_data = None
		for script in soup.find_all('script', type='application/ld+json'):
			txt = script.string
			if not txt:
				continue
			
			try:
				payload = json.loads(txt)
				
				# Handle single object or list of objects
				items = payload if isinstance(payload, list) else [payload]
				
				for obj in items:
					if isinstance(obj, dict) and obj.get("@type") == "NewsArticle":
						json_ld_data = obj
						break
				if json_ld_data:
					break
			except json.JSONDecodeError:
				continue

		# Extract from JSON-LD if found
		if json_ld_data:
			out["title"] = json_ld_data.get("headline") or json_ld_data.get("name") or ""
			out["date"] = json_ld_data.get("datePublished") or json_ld_data.get("dateModified") or ""
			out["content"] = json_ld_data.get("articleBody") or ""
			
			author_info = json_ld_data.get("author")
			if isinstance(author_info, dict):
				out["author"] = author_info.get("name", "")
			elif isinstance(author_info, list) and author_info:
				out["author"] = author_info[0].get("name", "")

		# ---------------- Fallback DOM Extraction ----------------
		if not out["content"]:
			# Fallback for title
			h1 = soup.select_one("div.content-column h1, h1")
			if h1:
				out["title"] = out["title"] or h1.get_text(strip=True)

			# Fallback for author
			author_el = soup.select_one("div.editor .name, .by-author a, a.authorName, span.name")
			if author_el:
				out["author"] = out["author"] or author_el.get_text(strip=True)

			# Fallback for date
			time_el = soup.select_one("time.article-publish-and-updated-date, time")
			if time_el:
				dt = time_el.get("datetime")
				out["date"] = out["date"] or (dt or time_el.get_text(strip=True))

			# Fallback for content
			paragraphs = soup.select("div.content-body.article-content p, article p, main p")
			content_texts = []
			for p in paragraphs:
				txt = p.get_text(strip=True)
				if txt and not txt.lower().startswith("advertisement"):
					content_texts.append(txt)
			out["content"] = " ".join(content_texts)

		# Final cleanup
		for k in out:
			if isinstance(out[k], str):
				out[k] = out[k].strip()

	except RequestException as e:
		print(f" 	⚠️ Error fetching article {out['url']}: HTTP/Connection Error ({e})")
	except Exception as e:
		print(f" 	⚠️ Error extracting article {out['url']}: Parsing Error ({e})")

	return out


# =====================================================
# Main Scraping Loop (Rewritten for requests)
# =====================================================

def main():
	start_time = time.time()
	scraped_urls, existing_df = load_existing_urls()
	print(f"🔁 Resuming... {len(scraped_urls)} URLs already scraped will be skipped.")

	buffer = []
	pages_processed = 0

	# Create a persistent session and headers for all requests
	session = requests.Session()

	for page_num in range(START_PAGE, END_PAGE + 1):
		headers = get_session_headers()
		wait_before = random.uniform(*DELAY_LISTING)
		print(f"\n🔎 Fetching listing page {page_num} (sleep {wait_before:.1f}s)")
		time.sleep(wait_before) # Use time.sleep for non-async environment

		listing_url = LISTING_TEMPLATE.format(page_num)
		
		try:
			# Fetch listing page (which returns JSON)
			listing_response = session.get(listing_url, headers=headers, timeout=60)
			listing_response.raise_for_status()

			# Content should be JSON as determined by previous steps
			data = listing_response.json()
			items = data.get("items") or data.get("page", {}).get("items") or []
			
			if not items:
				print(f"⚠️ No items in JSON on page {page_num}, skipping.")
				continue

			print(f"📰 Found {len(items)} items on page {page_num}")

			for item in items:
				href = item.get("href")
				title_text = item.get("titleText", "").strip()
				open_date = item.get("openDate", "")
				article_id = str(item.get("id") or "").strip()

				if not href:
					continue

				# --- URL CONSTRUCTION ---
				full_url = href if href.startswith("http") else BASE_DOMAIN + href
				
				if article_id and not re.search(r"/\d+/?$", full_url.rstrip("/")):
					full_url = full_url.rstrip("/") + f"/{article_id}"
					
				# REMOVED: print(f"🧩 Resolved article URL: {full_url}")

				if full_url in scraped_urls:
					continue

				# Random polite delay
				wait_a = random.uniform(*DELAY_ARTICLE)
				time.sleep(wait_a)

				# Extract article data using the requests function
				article_data = extract_article(full_url, headers)
				
				# Overlay non-parsed data just in case extraction failed
				article_data["title"] = article_data.get("title") or title_text
				article_data["date"] = article_data.get("date") or open_date
				# The canonical URL is captured inside extract_article, so we trust that.

				buffer.append(article_data)
				scraped_urls.add(article_data["url"]) 
				print(f" 	 	✅ Collected: {article_data.get('title', '')[:100]}")

			pages_processed += 1

			if pages_processed % SAVE_INTERVAL_PAGES == 0 and buffer:
				print(f"\n💾 Auto-saving after {pages_processed} pages...")
				save_append_excel(existing_df, buffer)
				df_new = pd.DataFrame(buffer)
				existing_df = pd.concat([existing_df, df_new], ignore_index=True)
				existing_df.drop_duplicates(subset="url", inplace=True)
				buffer = []

			if pages_processed % BACKUP_EVERY_PAGES == 0:
				backup_output()

		except RequestException as e:
			print(f"⚠️ Error fetching listing page {page_num}: HTTP/Connection Error ({e})")
		except json.JSONDecodeError:
			print(f"⚠️ Error processing listing page {page_num}: Content was not valid JSON.")
		except Exception as e:
			print(f"⚠️ Unexpected error processing listing page {page_num}: {e}")
			
	if buffer:
		print("\n💾 Final save of remaining articles...")
		save_append_excel(existing_df, buffer)

	elapsed = time.time() - start_time
	print(f"\n✅ Done scraping {START_PAGE} → {END_PAGE}")
	print(f"⏱ Total runtime: {elapsed:.2f}s ({elapsed/60:.2f} min)")
	print(f"📦 Total articles saved: {len(scraped_urls)}")


if __name__ == "__main__":
	main()
