const { chromium } = require("playwright");
const path = require("path");

const base = process.env.ATLAS_DEMO_URL || "http://127.0.0.1:8766";
const output = path.join(__dirname, "screenshots");

async function ready(page, url) {
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__ATLAS_READY === true, null, { timeout: 30000 });
  await page.waitForTimeout(250);
  if (errors.length) throw new Error(errors.join("\n"));
}

async function assertVisualIntegrity(page) {
  const result = await page.evaluate(() => {
    const canvases = [...document.querySelectorAll("canvas")].map((canvas) => ({
      id: canvas.id,
      width: canvas.width,
      height: canvas.height,
      cssWidth: canvas.clientWidth,
      cssHeight: canvas.clientHeight,
    }));
    const overflow = [...document.querySelectorAll("main *, .topbar *")]
      .filter((node) => node.children.length === 0 && node.textContent.trim())
      .filter((node) => node.scrollWidth > node.clientWidth + 2)
      .map((node) => ({
        tag: node.tagName,
        id: node.id,
        className: node.className,
        text: node.textContent.trim().slice(0, 80),
      }));
    return { canvases, overflow };
  });
  if (result.canvases.length !== 3) {
    throw new Error(`Expected 3 canvases, found ${result.canvases.length}`);
  }
  for (const canvas of result.canvases) {
    if (canvas.width < 600 || canvas.height < 360 || canvas.cssWidth < 250 || canvas.cssHeight < 180) {
      throw new Error(`Invalid canvas dimensions: ${JSON.stringify(canvas)}`);
    }
  }
  const actionable = result.overflow.filter((item) => !["run-id", "checkpoint-hash"].includes(item.id));
  if (actionable.length) throw new Error(`Text overflow: ${JSON.stringify(actionable)}`);
}

async function captureDataset(page, key, sample) {
  await ready(page, `${base}/?dataset=${key}&sample=${sample}`);
  await assertVisualIntegrity(page);
  await page.screenshot({
    path: path.join(output, `atlas_deployment_console_${key}.png`),
    fullPage: true,
  });
  await page.locator("#explainability").screenshot({
    path: path.join(output, `atlas_explainability_overview_${key}.png`),
  });
  await page.locator(".evidence-grid").evaluate((element) => {
    element.style.gridTemplateColumns = "1fr";
    element.style.maxWidth = "1200px";
    element.style.margin = "0 auto";
  });
  await page.evaluate(() => window.dispatchEvent(new Event("resize")));
  await page.waitForTimeout(150);
  const panels = [
    ["feature_evidence", "#feature-evidence-panel"],
    ["branch_attention", "#attention-panel"],
    ["faithfulness", "#faithfulness-panel"],
    ["calibration", "#calibration-panel"],
  ];
  for (const [name, selector] of panels) {
    await page.locator(selector).screenshot({
      path: path.join(output, `atlas_${name}_${key}.png`),
    });
  }
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 2,
      colorScheme: "light",
    });
    const page = await context.newPage();
    await captureDataset(page, "cic", 42);
    await captureDataset(page, "nf", 17);
    await page.locator(".deployment-band").screenshot({
      path: path.join(output, "atlas_compact_profile_nf.png"),
    });
    await context.close();

    const mobile = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 2,
      colorScheme: "light",
    });
    const mobilePage = await mobile.newPage();
    await ready(mobilePage, `${base}/?dataset=cic&sample=42`);
    const mobileOverflow = await mobilePage.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (mobileOverflow > 2) throw new Error(`Mobile horizontal overflow: ${mobileOverflow}px`);
    await mobilePage.screenshot({
      path: path.join(output, "qa_mobile.png"),
      fullPage: true,
    });
    await mobile.close();
    console.log("PASS: desktop/mobile rendering, canvas sizing, overflow, and screenshots");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
