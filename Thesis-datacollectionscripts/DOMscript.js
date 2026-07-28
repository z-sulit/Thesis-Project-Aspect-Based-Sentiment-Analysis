console.log('review scraper - Last ver najud');
window.stopScraping = null;

(async function() {
  const SCAN_INTERVAL_MS = 1000;
  const seen = new Set();
  const rows = [];
  let isRunning = true;

  // const titleFallback = document.title.split('-')[0].trim();
  const titleFallback = document.title.replace(/\s*[-–]\s*Google Maps\s*$/i, '').trim();
  const placeName = document.querySelector('h1.DUwDvf')?.innerText || titleFallback || 'Unknown Place';

  const coordsMatch = window.location.href.match(/@(-?\d+\.\d+),(-?\d+\.\d+)/);
  const lat = coordsMatch ? coordsMatch[1] : '';
  const lng = coordsMatch ? coordsMatch[2] : '';
  console.log(`Starting scrape for "${placeName}".`);

  const cleanText = (raw) => (raw || '')
    .replace(/[\r\n]+/g, ' ')
    .replace(/[\t\v\f\u200B-\u200D\uFEFF]/g, ' ')
    .replace(/[\x00-\x1F\x7F-\x9F]/g, '')
    .replace(/[\uE000-\uF8FF]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  const OWNER_REPLY_PATTERN = /response from (the )?owner/i;
  const isOwnerReplyText = (el) => {
    let node = el;
    for (let i = 0; i < 5 && node; i++) {
      const label = node.getAttribute?.('aria-label') || '';
      const prevSiblingText = node.previousElementSibling?.innerText || '';
      if (OWNER_REPLY_PATTERN.test(label) || OWNER_REPLY_PATTERN.test(prevSiblingText)) return true;
      node = node.parentElement;
    }
    return false;
  };

  const scrapeCurrentView = () => {
    const moreBtns = Array.from(document.querySelectorAll('button')).filter(btn => {
      const text = (btn.innerText || '').trim();
      const aria = (btn.getAttribute('aria-label') || '').trim();
      return text === 'More' || aria.includes('See more');
    });
    if (moreBtns.length > 0) moreBtns.forEach(btn => btn.click());

    const isReviewStar = (el) => {
      const label = (el.getAttribute('aria-label') || '').trim();
      return /^[1-5]\s*stars?$/i.test(label);
    };
    const starEls = Array.from(document.querySelectorAll('[role="img"][aria-label]')).filter(isReviewStar);

    starEls.forEach(starEl => {
      let node = starEl;
      let reviewContainer = node;
      for (let i = 0; i < 9 && node.parentElement; i++) {
        node = node.parentElement;
        const starsInside = Array.from(node.querySelectorAll('[role="img"][aria-label]')).filter(isReviewStar);
        if (starsInside.length === 1) {
          reviewContainer = node;
        } else {
          break;
        }
      }

      const rating = starEl.getAttribute('aria-label').replace(/[^0-9]/g, '');

      // Pick the first .wiI7pd that is NOT part of an owner's reply
      const candidateTextNodes = Array.from(reviewContainer.querySelectorAll('.wiI7pd'));
      const customerTextNode = candidateTextNodes.find(n => !isOwnerReplyText(n));
      const text = cleanText(customerTextNode ? customerTextNode.innerText : '');

      const containerCleaned = cleanText(reviewContainer.innerText || '');
      const fingerprint = containerCleaned.replace(/\s+/g, '').substring(0, 100) + text.substring(0, 50);

      if (!seen.has(fingerprint)) {
        seen.add(fingerprint);
        rows.push({ rating, text });
        console.log(`Grabbed new review! (Total: ${rows.length})`);
      }
    });
  };

  window.stopScraping = () => {
    isRunning = false;
    console.log(`Stopping... Exporting a total of ${rows.length} reviews.`);

    let csv = 'place_name,latitude,longitude,rating,review_text\n';
    rows.forEach(r => {
      const safeText = '"' + r.text.replace(/"/g, '""') + '"';
      const safePlaceName = '"' + placeName.replace(/"/g, '""') + '"';
      csv += `${safePlaceName},${lat},${lng},${r.rating},${safeText}\n`;
    });

    const BOM = '\uFEFF';
    const blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8' });

    const safeFileName = placeName
      .trim()
      .replace(/[\/\\:*?"<>|]/g, '')
      .replace(/\s+/g, '_');

    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${safeFileName}_reviews.csv`;
    a.click();
  };

  while (isRunning) {
    scrapeCurrentView();
    await new Promise(resolve => setTimeout(resolve, SCAN_INTERVAL_MS));
  }
})();

// Run stopScraping() to stop dom scrape (remove // then ctrl + enter)
// stopScraping()