const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = __dirname;
const diagnostics = process.env.REPORT_DIAGNOSTICS || path.join(root, '.report_check');

(async () => {
  fs.mkdirSync(diagnostics, {recursive: true});
  const browser = await chromium.launch({
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    // Raster copies make diagram lettering portable in GitHub and PDF readers.
    for (const name of ['architecture', 'retrieval_principle', 'task_lifecycle', 'u2_preprocessing']) {
      const file = path.join(root, 'figures', name + '.svg');
      const target = path.join(root, 'figures', name + '.png');
      if (!fs.existsSync(target) || process.argv.includes('--figures-only')) {
        const svg = fs.readFileSync(file, 'utf8');
        const width = Number(svg.match(/\bwidth="(\d+)"/)[1]);
        const height = Number(svg.match(/\bheight="(\d+)"/)[1]);
        const figure = await browser.newPage({viewport: {width, height}, deviceScaleFactor: 2});
        await figure.goto(pathToFileURL(file).href);
        await figure.evaluate(() => document.fonts.ready);
        await figure.screenshot({path: target});
        await figure.close();
      }
    }
    if (process.argv.includes('--figures-only')) {
      console.log(JSON.stringify({status: 'diagram_pngs_ready', count: 4}));
      return;
    }
    const page = await browser.newPage({viewport: {width: 1440, height: 1120}, deviceScaleFactor: 1});
    const errors = [];
    const external = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (/^https?:/.test(request.url())) external.push(request.url()); });
    await page.goto(pathToFileURL(path.join(root, 'technical_report.html')).href);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForFunction(() => [...document.images].filter(im => im.hasAttribute('src')).every(im => im.complete && im.naturalWidth > 0));
    const layout = await page.evaluate(() => [...document.querySelectorAll('.page')].map(section => {
      const content = section.querySelector('.content');
      const footer = section.querySelector('footer');
      const box = section.getBoundingClientRect();
      const c = content?.getBoundingClientRect();
      const f = footer?.getBoundingClientRect();
      return {id: section.id, page: Number(section.dataset.page), height: box.height,
        contentBottom: c?.bottom - box.top, footerTop: f?.top - box.top,
        overflow: content ? c.bottom > f.top - 7 : false,
        horizontal: section.scrollWidth > section.clientWidth + 1};
    }));
    fs.writeFileSync(path.join(diagnostics, 'layout_check.json'), JSON.stringify({errors, layout}, null, 2));
    await page.screenshot({path: path.join(diagnostics, 'preview_desktop.png')});
    if (layout.some(item => item.overflow || item.horizontal)) {
      throw Error('Page layout overflow: ' + layout.filter(item => item.overflow || item.horizontal).map(item => item.id).join(','));
    }
    await page.emulateMedia({media: 'print'});
    await page.pdf({path: path.join(root, 'technical_report.pdf'), format: 'A4', printBackground: true,
      preferCSSPageSize: true, displayHeaderFooter: false, tagged: true, outline: true});
    await page.emulateMedia({media: 'screen'});
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(diagnostics, 'preview_mobile.png')});
    const mobile = await page.evaluate(() => ({width: innerWidth, scrollWidth: document.documentElement.scrollWidth}));
    await page.selectOption('#jump', 'performance_latest');
    await page.waitForTimeout(500);
    await page.locator('#performance_latest figure img').first().click();
    const zoom = await page.locator('dialog').evaluate(dialog => dialog.open);
    await page.locator('dialog button').click();
    const result = {status: !errors.length && !external.length && mobile.scrollWidth <= mobile.width && zoom ? 'passed' : 'failed',
      pages: layout.length, imagesLoaded: true, desktopOverflow: false, mobile, zoom, errors, externalRequests: external};
    fs.writeFileSync(path.join(diagnostics, 'render_summary.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
    if (result.status !== 'passed') throw Error('Report browser checks failed');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
