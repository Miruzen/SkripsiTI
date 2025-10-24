import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import random
import time
import os
from datetime import datetime
import shutil

# === Configuration ===
OUTPUT_XLSX = "output/investing_articles_uji.xlsx"
BACKUP_FOLDER = "output/backups"
START_PAGE = 11
END_PAGE = 260
SAVE_INTERVAL = 5  # Save every N pages
BACKUP_INTERVAL = 4  # Create a backup every 4 saves


# === Helper Functions ===
def ensure_directories():
    os.makedirs("output", exist_ok=True)
    os.makedirs(BACKUP_FOLDER, exist_ok=True)


def backup_file():
    """Create timestamped backup of the main Excel file."""
    if os.path.exists(OUTPUT_XLSX):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.xlsx")
        shutil.copy2(OUTPUT_XLSX, backup_path)
        print(f"🗂️ Backup created → {backup_path}")


def save_progress(new_articles, save_count):
    """Safely append new articles and save Excel without losing data."""
    df_new = pd.DataFrame(new_articles)

    if os.path.exists(OUTPUT_XLSX):
        df_existing = pd.read_excel(OUTPUT_XLSX)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.drop_duplicates(subset="url", inplace=True)
    df_combined.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")

    print(f"💾 Saved {len(df_combined)} total articles → {OUTPUT_XLSX}")

    # Create backup every Nth save
    if save_count % BACKUP_INTERVAL == 0:
        backup_file()


async def scrape_single_article(context, url):
    """Scrape a single article page (title, author, date, content)."""
    data = {"author": "", "date": "", "content": ""}

    try:
        page = await context.new_page()
        await page.goto(url, timeout=90000, wait_until="domcontentloaded")

        # Handle redirect
        if "investing.com" in page.url and "/news/" not in page.url:
            print(f"⚠️ Redirected to homepage — retrying {url}")
            await asyncio.sleep(8)
            await page.goto(url, timeout=90000)

        await page.wait_for_selector("div#article", timeout=20000)

        author_el = await page.query_selector('a[data-test="article-provider-link"]')
        date_el = await page.query_selector('time[data-test="article-publish-date"]')

        if author_el:
            data["author"] = (await author_el.inner_text()).strip()
        if date_el:
            data["date"] = (await date_el.get_attribute("datetime")) or ""

        paragraphs = await page.query_selector_all("div#article p")
        text_parts = []
        for p in paragraphs:
            t = (await p.inner_text()).strip()
            if t and not t.lower().startswith("advertisement"):
                text_parts.append(t)
        data["content"] = " ".join(text_parts)

        await page.close()
    except Exception as e:
        print(f"  ⚠️ Error scraping {url}: {e}")
    return data


async def scrape_investing_articles(start_page=START_PAGE, end_page=END_PAGE, delay_range=(3, 7)):
    ensure_directories()
    start_time = time.time()

    scraped_urls = set()
    if os.path.exists(OUTPUT_XLSX):
        existing_df = pd.read_excel(OUTPUT_XLSX)
        scraped_urls = set(existing_df["url"].dropna().tolist())
        print(f"🔁 Resuming from {len(scraped_urls)} previously saved articles.")
    else:
        existing_df = pd.DataFrame(columns=["author", "date", "title", "url", "content"])

    base_url = "https://www.investing.com/currencies/eur-usd-news/"
    new_articles = []
    save_counter = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--start-maximized",
            ],
        )

        context = await browser.new_context(
            viewport={
                "width": random.randint(1280, 1920),
                "height": random.randint(720, 1080),
            },
            user_agent=random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 "
                "(KHTML, like Gecko) Version/16 Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/119 Safari/537.36",
            ]),
            java_script_enabled=True,
            locale="en-US",
        )

        page = await context.new_page()

        for i in range(start_page, end_page + 1):
            list_url = f"{base_url}{i}"
            print(f"\n🔎 Loading listing page {i}: {list_url}")

            try:
                await page.goto(list_url, timeout=90000, wait_until="domcontentloaded")

                if "eur-usd-news" not in page.url:
                    print("⚠️ Redirected by Cloudflare — retrying...")
                    await asyncio.sleep(8)
                    await page.goto(list_url, timeout=90000)

                await page.wait_for_selector('a[data-test="article-title-link"]', timeout=20000)
            except Exception as e:
                print(f"⚠️ Failed to load page {i}: {e}")
                continue

            article_cards = await page.query_selector_all("article")
            print(f"  Found {len(article_cards)} articles on page {i}")

            for card in article_cards:
                try:
                    link_el = await card.query_selector('a[data-test="article-title-link"]')
                    date_el = await card.query_selector("time")

                    if not link_el:
                        continue

                    href = await link_el.get_attribute("href")
                    title = (await link_el.inner_text()).strip()
                    date = (await date_el.inner_text()).strip() if date_el else ""

                    full_url = href if href.startswith("http") else f"https://www.investing.com{href}"

                    if full_url in scraped_urls:
                        continue

                    article_data = await scrape_single_article(context, full_url)
                    article_data["title"] = title
                    article_data["url"] = full_url
                    if not article_data.get("date"):
                        article_data["date"] = date

                    new_articles.append(article_data)
                    scraped_urls.add(full_url)

                    print(f"  ✅ Scraped: {title[:70]}... ({date})")
                    await asyncio.sleep(random.uniform(*delay_range))

                except Exception as e:
                    print(f"  ⚠️ Error processing article: {e}")
                    continue

            # === Auto-save every N pages ===
            if i % SAVE_INTERVAL == 0 and new_articles:
                save_counter += 1
                save_progress(new_articles, save_counter)
                new_articles.clear()
                print(f"💾 Auto-saved progress at page {i}.")

        # Final save
        if new_articles:
            save_progress(new_articles, save_counter + 1)

        await browser.close()

    duration = time.time() - start_time
    print(f"\n✅ Scraping completed from page {start_page} → {end_page}")
    print(f"⏱️ Total runtime: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"🧾 Total unique articles scraped: {len(scraped_urls)}")


if __name__ == "__main__":
    asyncio.run(scrape_investing_articles())
