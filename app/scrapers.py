import httpx
from bs4 import BeautifulSoup
import json
import re
from typing import List, Optional
from pydantic import BaseModel
import asyncio

# Define models here or import from schemas if available
class ScrapedNewsItem(BaseModel):
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[str] = None
    image_url: Optional[str] = None
    prediction_label: Optional[str] = None
    prediction_confidence: Optional[float] = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

class BaseScraper:
    def __init__(self, base_url: str, source_name: str):
        self.base_url = base_url
        self.source_name = source_name
        self.client = httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=10.0)

    async def fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = await self.client.get(url)
            if resp.status_code == 200:
                return resp.text
            print(f"Failed to fetch {url}: {resp.status_code}")
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def close(self):
        await self.client.aclose()

    async def get_news_links(self) -> List[str]:
        raise NotImplementedError

    async def scrape_article(self, url: str) -> Optional[ScrapedNewsItem]:
        raise NotImplementedError

class DailynewsScraper(BaseScraper):
    def __init__(self):
        super().__init__("https://www.dailynews.co.th/", "Dailynews")

    async def get_news_links(self) -> List[str]:
        html = await self.fetch_html("https://www.dailynews.co.th/news/")
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        # Pattern: /news/{digit}/
        for a in soup.find_all('a', href=True):
            href = a['href']
            if re.search(r'/news/\d+/', href):
                if href.startswith('/'):
                    href = "https://www.dailynews.co.th" + href
                links.add(href)
        return list(links)

    async def scrape_article(self, url: str) -> Optional[ScrapedNewsItem]:
        html = await self.fetch_html(url)
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        
        # Title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # Content
        # Prioritize potential content containers
        content_div = soup.select_one('.elementor-widget-theme-post-content')
        if not content_div:
            content_div = soup.select_one('#news-content')
        
        content = content_div.get_text(strip=True) if content_div else ""
        
        if not title or not content:
            print(f"Missing title or content for {url}")
            return None

        return ScrapedNewsItem(
            title=title,
            content=content,
            url=url,
            source=self.source_name
        )

class ThairathScraper(BaseScraper):
    def __init__(self):
        super().__init__("https://www.thairath.co.th", "Thairath")

    async def get_news_links(self) -> List[str]:
        html = await self.fetch_html("https://www.thairath.co.th/news/local") # Focus on local or main news
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        # Thairath structure: /news/local/..., /news/politic/...
        # Look for links starting with /news/
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/news/') and not href == '/news':
               # Filter out non-article links if possible (usually articles have /news/{category}/{id})
               if re.search(r'/news/\w+/\d+', href):
                    full_url = "https://www.thairath.co.th" + href
                    links.add(full_url)
        return list(links)

    async def scrape_article(self, url: str) -> Optional[ScrapedNewsItem]:
        html = await self.fetch_html(url)
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        
        item = ScrapedNewsItem(title="", content="", url=url, source=self.source_name)
        
        # Try JSON-LD first
        scripts = soup.find_all('script', type='application/ld+json')
        found_json = False
        for s in scripts:
            try:
                data = json.loads(s.string)
                if isinstance(data, list):
                     for d in data:
                         if d.get('@type') == 'NewsArticle':
                             item.title = d.get('headline', '')
                             item.content = d.get('articleBody', '')
                             item.published_date = d.get('datePublished')
                             found_json = True
                             break
                elif isinstance(data, dict):
                    if data.get('@type') == 'NewsArticle':
                         item.title = data.get('headline', '')
                         item.content = data.get('articleBody', '')
                         item.published_date = data.get('datePublished')
                         found_json = True
            except:
                pass
            if found_json: 
                break
        
        if not found_json or not item.title:
            # Fallback
            h1 = soup.find('h1')
            if h1:
                item.title = h1.get_text(strip=True)
            article = soup.find('article')
            if article:
                item.content = article.get_text(strip=True)

        if not item.title:
            return None
            
        return item

class ThaiPBSScraper(BaseScraper):
    def __init__(self):
        super().__init__("https://www.thaipbs.or.th", "ThaiPBS")

    async def get_news_links(self) -> List[str]:
        html = await self.fetch_html("https://www.thaipbs.or.th/news")
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        # Links matching /news/content/{id}
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/news/content/' in href:
                if href.startswith('/'):
                    href = "https://www.thaipbs.or.th" + href
                links.add(href)
        return list(links)

    async def scrape_article(self, url: str) -> Optional[ScrapedNewsItem]:
        html = await self.fetch_html(url)
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        
        item = ScrapedNewsItem(title="", content="", url=url, source=self.source_name)
        
        # Try __NEXT_DATA__
        script = soup.find('script', id='__NEXT_DATA__')
        if script:
            try:
                data = json.loads(script.string)
                page_props = data.get('props', {}).get('pageProps', {})
                content_data = page_props.get('content', {})
                if not content_data:
                     # fallback to rawContent?
                     content_data = page_props.get('rawContent', {})

                if content_data:
                    item.title = content_data.get('title', '')
                    # Content is often HTML in 'body' field or similar
                    # Check for 'body' or 'description' or 'detail'
                    # Based on debugging, we didn't see the exact keys inside 'content', but let's guess standard CMS keys
                    # If 'body' contains HTML, we might need to strip tags.
                    # Alternatively, 'content' might be a string if flat.
                    # As a backup, we can assume 'title' is key.
                    
                    # Inspect dictionary keys if possible (blindly here)
                    # Let's try 'detail' or 'body'
                    possible_content_keys = ['body', 'detail', 'content', 'description']
                    for k in possible_content_keys:
                        if k in content_data and content_data[k]:
                             # If it's HTML, clean it
                             raw_html = content_data[k]
                             text = BeautifulSoup(raw_html, 'html.parser').get_text(strip=True)
                             item.content = text
                             break
                    
                    if not item.content:
                         # Fallback if no content found in keys
                         item.content = str(content_data) # Ugly but ensures we get something

            except Exception as e:
                print(f"Error parsing Next data for {url}: {e}")

        if not item.title:
             h1 = soup.find('h1')
             if h1:
                 item.title = h1.get_text(strip=True)
        
        return item

class DisasterGoThScraper(BaseScraper):
    def __init__(self):
        super().__init__("https://www.disaster.go.th", "DisasterGoTh")

    async def get_news_links(self) -> List[str]:
        # Note: disaster.go.th is a Single Page Application (Nuxt.js).
        # Content is rendered client-side or fetched via internal API.
        # Without a browser or known API endpoint, we cannot easily scrape the map citations.
        # We will attempt to fetch the home page, but expect limited results in this environment.
        print(f"Fetching {self.base_url}/home ...")
        html = await self.fetch_html(f"{self.base_url}/home")
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        # Try to find any direct links
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Filter for likely news links (this is a guess based on standard patterns)
            if '/news/' in href or '/detail/' in href:
                if href.startswith('/'):
                    href = self.base_url + href
                links.add(href)
        
        if not links:
            print("  No links found (likely SPA).")
            # In a real scenario with browser access, we would wait for selector.
            
        return list(links)

    async def scrape_article(self, url: str) -> Optional[ScrapedNewsItem]:
        html = await self.fetch_html(url)
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        
        item = ScrapedNewsItem(title="", content="", url=url, source=self.source_name)
        
        # Heuristic extraction
        h1 = soup.find('h1')
        if h1:
            item.title = h1.get_text(strip=True)
            
        # Content
        article = soup.find('article')
        if article:
            item.content = article.get_text(strip=True)
        else:
             # Fallback
             divs = soup.find_all('div', class_=re.compile(r'content|detail|body'))
             for d in divs:
                 if len(d.get_text(strip=True)) > 100:
                     item.content = d.get_text(strip=True)
                     break
        
        if not item.title:
            return None
            
        return item

# Unified Function
async def scrape_all_sources(limit: int = 5) -> List[ScrapedNewsItem]:
    scrapers = [DailynewsScraper(), ThairathScraper(), ThaiPBSScraper(), DisasterGoThScraper()]
    all_news = []
    
    for scraper in scrapers:
        print(f"Scraping {scraper.source_name}...")
        try:
            links = await scraper.get_news_links()
            print(f"Found {len(links)} links for {scraper.source_name}")
            
            # Limit scraping
            for link in links[:limit]:
                try:
                    print(f"  Scraping: {link}")
                    article = await scraper.scrape_article(link)
                    if article:
                        all_news.append(article)
                except Exception as e:
                    print(f"  Failed to scrape {link}: {e}")
        except Exception as e:
            print(f"Error scraping source {scraper.source_name}: {e}")
        finally:
            await scraper.close()
        
    return all_news
