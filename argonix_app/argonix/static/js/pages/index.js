// Registers the Analytics and Journeys pages (lazy-loaded) and their stylesheet. Imported once by app.js.
import { registerPage } from "../registry.js";

try {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("../../pages.css", import.meta.url).href;   // /static/pages.css
  document.head.appendChild(link);
} catch { /* styles are cosmetic; the pages still work */ }

registerPage({ id: "analytics", label: "Analytics", icon: "analytics", group: "monitor", order: 60, perm: "alert.read",
  load: () => import("./analytics.js"),
  keywords: ["counting", "lines", "flow", "density", "queue", "dwell", "counter-flow", "dispersal", "hotspots", "anomaly", "cusum"] });

registerPage({ id: "journeys", label: "Journeys", icon: "journeys", group: "monitor", order: 70, perm: "alert.read",
  load: () => import("./journeys.js"),
  keywords: ["re-id", "reid", "global id", "handoff", "cross-camera", "timeline", "path"] });
