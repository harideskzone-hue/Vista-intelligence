const fs = require("fs");
const { JSDOM } = require("jsdom");

const html = fs.readFileSync("face_api/templates/preview.html", "utf8");
const dom = new JSDOM(html, {
    url: "http://localhost:5001/preview",
    runScripts: "dangerously",
    resources: "usable"
});

// Mock fetch
dom.window.fetch = async (url) => {
    if (url === "/api/cameras") {
        return {
            ok: true,
            json: async () => ({
                success: true,
                cameras: [
                    { id: "cam_1", source: "1", label: "Cam 1", enabled: true }
                ]
            })
        };
    }
    if (url === "/api/health") {
        return {
            ok: true,
            json: async () => ({})
        };
    }
    if (url === "/api/camera/status") {
        return {
            ok: true,
            json: async () => ({ active: true, pending_jobs: 0 })
        };
    }
    return { ok: true, json: async () => ({}) };
};

dom.window.document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
        const grid = dom.window.document.getElementById("live-camera-grid");
        console.log("Grid innerHTML length:", grid ? grid.innerHTML.length : "null");
        if (grid && grid.innerHTML.trim() === "<!-- Dynamically populated -->") {
            console.log("Grid was not populated.");
        }
    }, 1000);
});

dom.window.onerror = function(msg, url, line, col, error) {
    console.log("JSDOM ERROR:", msg, line, col, error);
};

process.on("unhandledRejection", (err) => {
    console.log("UNHANDLED REJECTION:", err);
});
