// Apply stored display preferences (theme, demo mode) before first paint (no server data involved). A classic (non-module) script so it runs
// synchronously in <head>; it lives in a file because the Content-Security-Policy forbids inline scripts.
try {
  var t = localStorage.getItem("argonix.theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  if (localStorage.getItem("argonix.demo") === "1") document.documentElement.classList.add("demo");
} catch (e) { /* storage unavailable */ }
