"use client";

import { ChangeEvent, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

const categories = [
  { value: "bottle", label: "Bottle" },
  { value: "cable", label: "Cable" },
  { value: "capsule", label: "Capsule" },
  { value: "carpet", label: "Carpet" },
  { value: "grid", label: "Grid" },
  { value: "hazelnut", label: "Hazelnut" },
  { value: "leather", label: "Leather" },
  { value: "metal_nut", label: "Metal Nut" },
  { value: "pill", label: "Pill" },
  { value: "screw", label: "Screw" },
  { value: "tile", label: "Tile" },
  { value: "toothbrush", label: "Toothbrush" },
  { value: "transistor", label: "Transistor" },
  { value: "wood", label: "Wood" },
  { value: "zipper", label: "Zipper" },
];

const allowedExtensions = [
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".avif",
];

type InspectionResult = {
  category: string;
  anomaly_score: number;
  threshold: number;
  prediction: "NORMAL" | "ANOMALY";
  heatmap_url: string;
  explanation: string;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("bottle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ==========================================================
  // FILE SELECTION
  // ==========================================================

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    const fileName = selectedFile.name.toLowerCase();

    const validExtension = allowedExtensions.some(
      (extension) => fileName.endsWith(extension)
    );

    if (!validExtension) {
      setError(
        "Unsupported image format. Please upload JPG, JPEG, PNG, WEBP, or AVIF."
      );
      setFile(null);
      setPreviewUrl(null);
      setResult(null);
      return;
    }

    console.log("Selected file:", selectedFile.name);

    setFile(selectedFile);
    setError("");
    setResult(null);

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
  };

  // ==========================================================
  // INSPECT IMAGE
  // ==========================================================

  const inspectImage = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append("file", file);
    formData.append("category", category);

    try {
      const response = await fetch(`${API_URL}/inspect`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Inspection failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the inspection backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // CLEAR IMAGE
  // ==========================================================

  const clearImage = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");

    const fileInput = document.getElementById(
      "image-upload"
    ) as HTMLInputElement | null;

    if (fileInput) {
      fileInput.value = "";
    }
  };

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-6xl px-6 py-10">

        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <header className="mb-10">
          <p className="mb-2 text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
            AI Industrial Inspection
          </p>

          <h1 className="text-4xl font-bold tracking-tight">
            Visual Anomaly Detection
          </h1>

          <p className="mt-3 max-w-2xl text-slate-400">
            Upload an industrial product image to detect anomalies,
            generate a heatmap, and receive an AI inspection explanation.
          </p>
        </header>

        {/* ================================================== */}
        {/* MAIN GRID */}
        {/* ================================================== */}

        <div className="grid gap-8 lg:grid-cols-2">

          {/* ================================================= */}
          {/* UPLOAD CARD */}
          {/* ================================================= */}

          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">

            <h2 className="text-xl font-semibold">
              Upload Image
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Select the product category and upload your image.
            </p>

            {/* CATEGORY */}

            <div className="mt-6">
              <label
                htmlFor="category"
                className="mb-2 block text-sm font-medium text-slate-300"
              >
                Product Category
              </label>

              <select
                id="category"
                value={category}
                onChange={(event) => {
                  setCategory(event.target.value);
                  setResult(null);
                }}
                className="w-full rounded-xl border border-slate-700 bg-slate-800 px-4 py-3 text-white outline-none focus:border-cyan-400"
              >
                {categories.map((item) => (
                  <option
                    key={item.value}
                    value={item.value}
                  >
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            {/* FILE */}

            <div className="mt-6">
              <label
                htmlFor="image-upload"
                className="mb-2 block text-sm font-medium text-slate-300"
              >
                Product Image
              </label>

              <input
                id="image-upload"
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.avif,image/jpeg,image/png,image/webp,image/avif"
                onChange={handleFileChange}
                className="block w-full cursor-pointer rounded-xl border border-slate-700 bg-slate-800 p-3 text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:font-medium file:text-slate-950 hover:file:bg-cyan-400"
              />

              <p className="mt-2 text-xs text-slate-500">
                Supported formats: JPG, JPEG, PNG, WEBP, AVIF
              </p>
            </div>

            {/* SELECTED FILE */}

            {file && (
              <div className="mt-4 flex items-center justify-between rounded-xl border border-emerald-900 bg-emerald-950/20 px-4 py-3">
                <div className="min-w-0">
                  <p className="text-xs text-slate-500">
                    Selected file
                  </p>

                  <p className="truncate text-sm font-medium text-emerald-400">
                    {file.name}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={clearImage}
                  className="ml-4 rounded-lg px-3 py-1.5 text-xs text-slate-400 hover:bg-slate-800 hover:text-white"
                >
                  Remove
                </button>
              </div>
            )}

            {/* PREVIEW */}

            {previewUrl && (
              <div className="mt-6">
                <p className="mb-2 text-sm font-medium text-slate-300">
                  Preview
                </p>

                <div className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
                  <img
                    src={previewUrl}
                    alt="Selected product"
                    className="max-h-[380px] w-full object-contain"
                  />
                </div>
              </div>
            )}

            {/* INSPECT BUTTON */}

            <button
              type="button"
              onClick={inspectImage}
              disabled={loading}
              className="mt-6 w-full rounded-xl bg-cyan-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Inspecting..." : "Inspect Image"}
            </button>

            {/* ERROR */}

            {error && (
              <div className="mt-4 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
                {error}
              </div>
            )}

          </section>

          {/* ================================================= */}
          {/* RESULTS CARD */}
          {/* ================================================= */}

          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">

            <h2 className="text-xl font-semibold">
              Inspection Result
            </h2>

            {/* NO RESULT */}

            {!result && !loading && (
              <div className="flex min-h-[420px] items-center justify-center text-center text-slate-500">
                <div>
                  <p className="text-lg">
                    No inspection yet
                  </p>

                  <p className="mt-2 text-sm">
                    Upload an image and click
                    <br />
                    "Inspect Image".
                  </p>
                </div>
              </div>
            )}

            {/* LOADING */}

            {loading && (
              <div className="flex min-h-[420px] items-center justify-center text-center">
                <div>
                  <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-slate-700 border-t-cyan-400" />

                  <p className="mt-4 text-slate-400">
                    Running inspection...
                  </p>

                  <p className="mt-1 text-xs text-slate-600">
                    Anomaly detection and AI explanation are being generated.
                  </p>
                </div>
              </div>
            )}

            {/* RESULT */}

            {result && !loading && (
              <div className="mt-6">

                {/* ================================================= */}
                {/* PREDICTION */}
                {/* ================================================= */}

                <div
                  className={`rounded-2xl border p-5 ${
                    result.prediction === "ANOMALY"
                      ? "border-red-900 bg-red-950/30"
                      : "border-emerald-900 bg-emerald-950/30"
                  }`}
                >
                  <p className="text-sm text-slate-400">
                    Detection
                  </p>

                  <p
                    className={`mt-1 text-3xl font-bold ${
                      result.prediction === "ANOMALY"
                        ? "text-red-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {result.prediction}
                  </p>
                </div>

                {/* ================================================= */}
                {/* METRICS */}
                {/* ================================================= */}

                <div className="mt-5 grid grid-cols-2 gap-4">

                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Anomaly Score
                    </p>

                    <p className="mt-1 text-xl font-semibold">
                      {result.anomaly_score.toFixed(4)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Threshold
                    </p>

                    <p className="mt-1 text-xl font-semibold">
                      {result.threshold.toFixed(4)}
                    </p>
                  </div>

                </div>

                {/* ================================================= */}
                {/* HEATMAP */}
                {/* ================================================= */}

                <div className="mt-6">

                  <div className="mb-2 flex items-center justify-between">
                    <p className="text-sm font-medium text-slate-300">
                      Anomaly Heatmap
                    </p>

                    <span className="text-xs text-slate-500">
                      {result.category}
                    </span>
                  </div>

                  <div className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
                    <img
                      src={`${API_URL}${result.heatmap_url}`}
                      alt="Anomaly heatmap"
                      className="w-full object-contain"
                    />
                  </div>

                </div>

                {/* ================================================= */}
                {/* AI EXPLANATION */}
                {/* ================================================= */}

                <div className="mt-6 rounded-2xl border border-cyan-900/60 bg-cyan-950/20 p-5">

                  <div className="flex items-center gap-3">

                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-400">
                      ✦
                    </div>

                    <div>
                      <p className="text-sm font-semibold text-cyan-400">
                        AI Inspection Explanation
                      </p>

                      <p className="text-xs text-slate-500">
                        Generated from the inspected image
                      </p>
                    </div>

                  </div>

                  {/* ONLY THIS PART CHANGED */}

                  <div className="mt-5 text-sm leading-7 text-slate-300">

                    <div className="mb-5">
                      <p className="mb-1 font-semibold text-cyan-400">
                        Result
                      </p>

                      <p>
                        {result.explanation
                          .split("Observation:")[0]
                          .replace("Result:", "")
                          .trim()}
                      </p>
                    </div>

                    <div className="mb-5">
                      <p className="mb-1 font-semibold text-cyan-400">
                        Observation
                      </p>

                      <p>
                        {result.explanation
                          .split("Observation:")[1]
                          ?.split("Reason:")[0]
                          ?.trim()}
                      </p>
                    </div>

                    <div className="mb-5">
                      <p className="mb-1 font-semibold text-cyan-400">
                        Reason
                      </p>

                      <p>
                        {result.explanation
                          .split("Reason:")[1]
                          ?.split("Recommended Action:")[0]
                          ?.trim()}
                      </p>
                    </div>

                    <div>
                      <p className="mb-1 font-semibold text-cyan-400">
                        Recommended Action
                      </p>

                      <p>
                        {result.explanation
                          .split("Recommended Action:")[1]
                          ?.trim()}
                      </p>
                    </div>

                  </div>

                </div>

              </div>
            )}

          </section>

        </div>

        {/* ================================================== */}
        {/* FOOTER */}
        {/* ================================================== */}

        <footer className="mt-10 border-t border-slate-800 pt-6 text-center text-sm text-slate-600">
          Anomaly detection powered by AnomalyCLIP
        </footer>

      </div>
    </main>
  );
}