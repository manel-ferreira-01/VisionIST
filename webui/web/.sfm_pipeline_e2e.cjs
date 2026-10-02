/** Live check of the sfm_video pipeline page (real webui + lightglue, moge
 *  and sfm boxes): upload frames through the video_frames widget, Run,
 *  watch the steps, wait for the scene, screenshot.
 *    node .sfm_pipeline_e2e.cjs <frames dir> [out dir]
 *  (frames as JPEGs: headless_shell has no H.264, so no mp4 here) */
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const os = require('os');

const dir = process.argv[2];
const out = process.argv[3] || '/tmp';
(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
    args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1600 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

  await page.goto('http://127.0.0.1:8090/index.html#/pipeline/sfm_video', { waitUntil: 'networkidle' });
  await page.evaluate(() => sessionStorage.clear());
  await page.reload({ waitUntil: 'networkidle' });
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.jpg')).sort().map((f) => path.join(dir, f));
  await page.setInputFiles('input[type="file"]', files);
  await page.waitForFunction((n) => document.querySelectorAll('.filechip').length === n, files.length, { timeout: 60000 });
  await page.click('button:has-text("Run")');
  await page.waitForSelector('.steps .step', { timeout: 30000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: `${out}/sfm_pipeline_running.png`, fullPage: false });
  await page.waitForSelector('.resulthead .chip.ok', { timeout: 600000 });
  await page.waitForTimeout(5000);            // scene buffers + a few RAF frames
  const canvases = page.locator('.glbview canvas');
  console.log('canvases:', await canvases.count());
  console.log('steps:', (await page.locator('.steps .step').allTextContents()).join(' || '));
  await canvases.first().screenshot({ path: `${out}/sfm_pipeline_scene.png` });
  await page.screenshot({ path: `${out}/sfm_pipeline_page.png`, fullPage: true });
  // toggle "color depth by frame" and shoot again
  await page.locator('label:has-text("color depth by frame") input').check();
  await page.waitForTimeout(800);
  await canvases.first().screenshot({ path: `${out}/sfm_pipeline_scene_byframe.png` });
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
  if (errors.length) { console.error('FAIL: page errors'); process.exit(1); }
  console.log('OK');
})().catch((e) => { console.error('FATAL', e.message); process.exit(1); });
