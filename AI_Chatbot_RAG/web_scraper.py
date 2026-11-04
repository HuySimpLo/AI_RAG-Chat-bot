import json
import sys
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def scrape(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            body = soup.find("main") or soup.find("body") or soup
            return {
                "url": page.url,
                "text": body.get_text(separator=" ", strip=True),
                "html_length": len(html),
            }
        finally:
            browser.close()

if __name__ == "__main__":
    try:
        print(json.dumps(scrape(sys.argv[1]), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        sys.exit(1)
