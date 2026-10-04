// Use only browser-exposed information. Missing values remain unknown.
export function deviceType(browser) {
  const agent = browser.userAgent || ''
  if (/iPad|Tablet/i.test(agent) || /Macintosh/i.test(agent) && browser.maxTouchPoints > 1) return 'tablet'
  if (browser.userAgentData?.mobile || /Mobi|iPhone/i.test(agent)) return 'phone'
  if (/Android/i.test(agent)) return 'tablet'
  return 'laptop'
}

export function hardwareInfo(browser) {
  const cores = browser.hardwareConcurrency, memory = browser.deviceMemory;
  return {
    logical_cores: Number.isInteger(cores) && cores >= 1 && cores <= 4096 ? cores : null,
    memory_gib: Number.isFinite(memory) && memory > 0 && memory <= 65536 ? memory : null,
    platform: String(browser.userAgentData?.platform || browser.platform || '').slice(0, 80),
  };
}
