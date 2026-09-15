
        // ── Master Camera Toggle API ─────────────────────────────────────────
        let isCameraActive = false;
        async function fetchCameraStatus() {
            try {
                const res = await fetch('/api/camera/status');
                if (res.ok) {
                    const data = await res.json();
                    updateCameraUI(data.active, data.pending_jobs || 0);
                }
            } catch (e) { console.error('Failed to read camera state', e); }
        }

        function updateCameraUI(active, pending_jobs = 0) {
            isCameraActive = active;
            const btn = document.getElementById('cam-toggle-btn');
            const ind = document.getElementById('cam-status-indicator');
            const pipeInd = document.getElementById('pipeline-status-indicator');
            
            if (active) {
                btn.textContent = "Stop Camera";
                btn.classList.add('bg-red-500', 'text-white', 'hover:bg-red-600');
                btn.classList.remove('bg-primary', 'dark:bg-white', 'dark:text-primary');
                ind.innerHTML = `
                    <span class="relative flex h-2.5 w-2.5">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-20"></span>
                        <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]"></span>
                    </span>
                    <span class="text-[10px] font-bold text-green-500 uppercase tracking-widest">Camera Running</span>
                `;
            } else {
                btn.textContent = "Start Camera";
                btn.classList.remove('bg-red-500', 'text-white', 'hover:bg-red-600');
                btn.classList.add('bg-primary', 'dark:bg-white', 'dark:text-primary');
                ind.innerHTML = `
                    <span class="relative flex h-2.5 w-2.5">
                        <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-gray-400"></span>
                    </span>
                    <span class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Camera Stopped</span>
                `;
            }
            
            // Set dual-state Pipeline Indicator
            if (pending_jobs > 0) {
                pipeInd.classList.remove('hidden');
                // Use absolute positioning or margin for correct visual placement
                pipeInd.innerHTML = `
                    <div class="h-4 w-px bg-gray-300 dark:bg-white/20 mx-1"></div>
                    <span class="relative flex h-2.5 w-2.5">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-500 opacity-40"></span>
                        <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500"></span>
                    </span>
                    <span class="text-[10px] font-bold text-amber-500 uppercase tracking-widest">Processing ${pending_jobs} Job${pending_jobs>1?'s':''}...</span>
                `;
            } else {
                pipeInd.classList.remove('hidden');
                pipeInd.innerHTML = `
                    <div class="h-4 w-px bg-gray-300 dark:bg-white/20 mx-1"></div>
                    <span class="relative flex h-2.5 w-2.5">
                        <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-gray-300 dark:bg-white/20"></span>
                    </span>
                    <span class="text-[10px] font-bold text-gray-400 dark:text-white/40 uppercase tracking-widest">Pipeline Idle</span>
                `;
            }
        }

        async function toggleSystemCamera() {
            const newState = !isCameraActive;
            const endpoint = newState ? '/api/camera/start' : '/api/camera/stop';
            updateCameraUI(newState); // Optimistic UI update
            try {
                const res = await fetch(endpoint, {method: 'POST', credentials: 'same-origin'});
                if (!res.ok) throw new Error("API Failure");
            } catch (e) {
                updateCameraUI(!newState); // Revert on failure
                console.error("Failed to toggle camera", e);
            }
        }

        fetchCameraStatus();
        setInterval(fetchCameraStatus, 1500); // Poll status
    