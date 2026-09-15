const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
    const page = await browser.newPage();
    
    page.on('console', msg => {
        if (msg.type() === 'error') {
            console.log('BROWSER ERROR:', msg.text());
        }
    });
    
    page.on('pageerror', err => {
        console.log('PAGE ERROR:', err.toString());
    });

    try {
        await page.goto('http://localhost:5001/preview', { waitUntil: 'networkidle0', timeout: 5000 });
        console.log('Page loaded.');
    } catch (e) {
        console.log('Navigation error:', e.toString());
    }
    
    await browser.close();
})();
