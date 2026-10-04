// Use only browser-exposed information. Missing values remain unknown.
export function hardwareInfo(browser) {
  const cores = browser.hardwareConcurrency, memory = browser.deviceMemory;
  return {
    logical_cores: Number.isInteger(cores) && cores >= 1 && cores <= 4096 ? cores : null,
    memory_gib: Number.isFinite(memory) && memory > 0 && memory <= 65536 ? memory : null,
    platform: String(browser.userAgentData?.platform || browser.platform || '').slice(0, 80),
  };
}
