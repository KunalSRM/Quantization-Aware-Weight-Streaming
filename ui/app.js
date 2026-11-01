
const $ = (id) => document.getElementById(id);

let latencyChart = null;
let precisionChart = null;
let throughputChart = null;
let memoryChart = null;
let lastResults = [];

const VALID_MODELS = [
  "qwen_0_5b", "qwen_1_5b", "qwen_3b",
  "llama_1b", "llama_3b",
  "smollm_135m", "smollm_360m", "smollm_1_7b",
  "gemma_270m", "gemma_1b", "gemma_4b"
];

const VALID_PRECISIONS = ["FP16", "INT8", "INT4", "INT3", "INT2"];

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }

  return data;
}

function setMessage(message, isError = false) {
  const element = $("message");
  if (!element) return;

  element.textContent = message;
  element.style.color = isError ? "#ff7777" : "";
}

function escapeHtml(value) {
  return String(value ?? "—").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";

  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

// Read the exact values from the dropdowns and send them to FastAPI.
async function startExperiment() {
  const model = $("model")?.value?.trim().toLowerCase();
  const precision = $("precision")?.value?.trim().toUpperCase();
  const button = $("runButton");

  if (!VALID_MODELS.includes(model)) {
    setMessage(`Invalid model ID: ${model}. Check the model dropdown in index.html.`, true);
    return;
  }

  if (!VALID_PRECISIONS.includes(precision)) {
    setMessage(`Invalid precision: ${precision}.`, true);
    return;
  }

  if (button) button.disabled = true;

  setMessage(`Submitting ${model} with ${precision}...`);

  try {
    // Log the exact values being sent for troubleshooting.
    console.log("QAWS experiment request:", { model, precision });

    const result = await api("/api/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: model,
        precision: precision
      })
    });

    setMessage(`Experiment started: ${result.model} · ${result.precision}`);
    await refreshStatus();
  } catch (error) {
    setMessage(`Could not start experiment: ${error.message}`, true);
    if (button) button.disabled = false;
  }
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");

    const connection = $("connection");
    const logOutput = $("logOutput");
    const logState = $("logState");
    const button = $("runButton");

    if (connection) connection.textContent = "Backend connected";

    if (logOutput) {
      logOutput.textContent = data.log || "No experiment started.";
      logOutput.scrollTop = logOutput.scrollHeight;
    }

    if (logState) {
      logState.textContent = String(data.status || "idle").toUpperCase();
    }

    if (button) {
      button.disabled = Boolean(data.running);
      button.textContent = data.running
        ? "Experiment running..."
        : "▶  Run experiment";
    }

    if (data.status === "completed") {
      setMessage("Experiment completed successfully.");
      await refreshResults();
    } else if (data.status === "failed") {
      setMessage("Experiment failed. Check Experiment logs below.", true);
    }
  } catch (error) {
    const connection = $("connection");
    if (connection) connection.textContent = "Backend disconnected";

    console.error("Status refresh failed:", error);
  }
}

function renderResults(results) {
  const body = $("resultsBody");
  if (!body) return;

  if (!results.length) {
    body.innerHTML =
      '<tr><td colspan="7" class="empty">No benchmark results found.</td></tr>';
    return;
  }

  body.innerHTML = results.map((item) => `
    <tr>
      <td>${escapeHtml(item.file)}</td>
      <td>${escapeHtml(item.model)}</td>
      <td>${escapeHtml(item.precision)}</td>
      <td>${formatNumber(item.latency)}</td>
      <td>${formatNumber(item.throughput)}</td>
      <td>${formatNumber(item.ram)}</td>
      <td>${formatNumber(item.gpu_memory)}</td>
    </tr>
  `).join("");
}

function average(items, key) {
  const values = items
    .map((item) => Number(item[key]))
    .filter((value) => Number.isFinite(value));

  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function destroyChart(chart) {
  if (chart) chart.destroy();
}

function renderCharts(results) {
  if (typeof Chart === "undefined") {
    const hint = $("chartHint");
    if (hint) hint.textContent = "CHART LIBRARY UNAVAILABLE";
    return;
  }

  const recent = [...results].reverse().slice(-15);

  const labels = recent.map((item) =>
    `${item.model || "Model"} · ${item.precision || "?"}`
  );

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false
  };

  // Latency line chart.
  destroyChart(latencyChart);
  latencyChart = new Chart($("latencyChart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Latency (seconds)",
        data: recent.map((item) => item.latency),
        borderColor: "#8b7cff",
        backgroundColor: "rgba(139,124,255,0.15)",
        fill: true,
        tension: 0.3,
        spanGaps: true
      }]
    },
    options: chartOptions
  });

  // Precision doughnut chart.
  const precisionCounts = {};

  results.forEach((item) => {
    const precision = item.precision || "Unknown";
    precisionCounts[precision] = (precisionCounts[precision] || 0) + 1;
  });

  destroyChart(precisionChart);
  precisionChart = new Chart($("precisionChart"), {
    type: "doughnut",
    data: {
      labels: Object.keys(precisionCounts),
      datasets: [{
        data: Object.values(precisionCounts),
        backgroundColor: [
          "#8b7cff", "#40c9a2", "#f5b85c", "#ef7185", "#5aa9ff"
        ]
      }]
    },
    options: {
      ...chartOptions,
      cutout: "65%"
    }
  });

  // Average throughput by precision.
  const precisions = [...new Set(
    results.map((item) => item.precision || "Unknown")
  )];

  const throughputValues = precisions.map((precision) =>
    average(
      results.filter((item) => (item.precision || "Unknown") === precision),
      "throughput"
    )
  );

  destroyChart(throughputChart);
  throughputChart = new Chart($("throughputChart"), {
    type: "bar",
    data: {
      labels: precisions,
      datasets: [{
        label: "Tokens per second",
        data: throughputValues,
        backgroundColor: "#40c9a2",
        borderRadius: 5
      }]
    },
    options: chartOptions
  });

  // RAM and GPU memory comparison.
  destroyChart(memoryChart);
  memoryChart = new Chart($("memoryChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "RAM (GB)",
          data: recent.map((item) => item.ram),
          backgroundColor: "#8b7cff"
        },
        {
          label: "GPU memory (GB)",
          data: recent.map((item) => item.gpu_memory),
          backgroundColor: "#40c9a2"
        }
      ]
    },
    options: {
      ...chartOptions,
      scales: {
        x: {
          ticks: {
            maxRotation: 60,
            minRotation: 20
          }
        }
      }
    }
  });
}

async function refreshResults() {
  try {
    const data = await api("/api/results");
    const results = data.results || [];

    lastResults = results;

    if ($("resultCount")) {
      $("resultCount").textContent = data.count ?? results.length;
    }

    if ($("tableCount")) {
      $("tableCount").textContent = `${results.length} RUNS`;
    }

    const latest = results[0];

    if (latest) {
      if ($("latestLatency")) {
        $("latestLatency").textContent = latest.latency == null
          ? "—"
          : `${formatNumber(latest.latency)} s`;
      }

      if ($("latestThroughput")) {
        $("latestThroughput").textContent = latest.throughput == null
          ? "—"
          : `${formatNumber(latest.throughput)} tok/s`;
      }

      if ($("latestGpu")) {
        $("latestGpu").textContent = latest.gpu_memory == null
          ? "—"
          : `${formatNumber(latest.gpu_memory)} GB`;
      }
    }

    renderResults(results);
    renderCharts(results);
  } catch (error) {
    console.error("Results refresh failed:", error);
    setMessage(`Could not load results: ${error.message}`, true);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("runButton")?.addEventListener("click", startExperiment);
  $("refreshButton")?.addEventListener("click", refreshResults);

  refreshStatus();
  refreshResults();

  setInterval(refreshStatus, 2000);
  setInterval(refreshResults, 10000);
});
