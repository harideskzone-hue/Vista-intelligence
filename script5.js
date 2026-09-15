
        // ── Camera Management Logic ───────────────────────────────────────────
        
        async function fetchCameras() {
            try {
                const res = await fetch('/api/cameras');
                if (!res.ok) return;
                const data = await res.json();
                renderCameras(data.cameras || []);
            } catch (e) { console.error('Failed to fetch cameras', e); }
        }

        function renderCameras(cameras) {
            const list = document.getElementById('cam-list');
            const badge = document.getElementById('cam-count-badge');
            badge.textContent = `${cameras.length} / 6 cameras`;
            
            if (cameras.length === 0) {
                list.innerHTML = `<div class="text-xs text-text-tertiary italic py-2">No cameras configured. Add up to 6 cameras (3 wired USB + 3 IP/RTSP) below.</div>`;
                return;
            }

            list.innerHTML = cameras.map(cam => `
                <div class="flex items-center justify-between p-3 rounded-lg border ${cam.enabled ? 'border-primary/20 bg-blue-50/30 dark:border-white/20 dark:bg-white/5' : 'border-gray-200 bg-gray-50 opacity-60 dark:border-white/10 dark:bg-white/5'} transition-all mb-2">
                    <div class="flex items-center gap-3">
                        <div class="${cam.enabled ? 'bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.4)]' : 'bg-gray-300'} rounded-full" style="width: 8px; height: 8px; min-width: 8px;"></div>
                        <div>
                            <p class="text-xs font-semibold text-text-primary dark:text-white">${cam.label || cam.id}</p>
                            <p class="text-[10px] text-text-tertiary font-mono mt-0.5">${cam.source}</p>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <button onclick="toggleCam('${cam.id}', ${!cam.enabled})" class="text-[10px] uppercase font-bold px-2 py-1 rounded bg-white dark:bg-white/10 border border-gray-200 dark:border-white/20 hover:border-primary dark:hover:border-white transition-colors text-text-primary dark:text-white">
                            ${cam.enabled ? 'Disable' : 'Enable'}
                        </button>
                        <button onclick="deleteCam('${cam.id}')" class="text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 p-1 rounded transition-colors material-symbols-outlined text-[18px]">
                            delete
                        </button>
                    </div>
                </div>
            `).join('');
        
            // ---- LIVE PREVIEW GRID POPULATION ----
            const previewGrid = document.getElementById('live-camera-grid');
            if (previewGrid) {
                if (cameras.length === 0) {
                    previewGrid.innerHTML = `<div class="col-span-full py-12 text-center text-text-tertiary bg-white dark:bg-dark-card rounded-2xl border border-gray-100 dark:border-white/5">No active cameras found. Add one in the dashboard.</div>`;
                } else {
                    const activeCams = cameras.filter(cam => cam.enabled);
                    if (activeCams.length === 0) {
                        previewGrid.innerHTML = `<div class="col-span-full py-12 text-center text-text-tertiary bg-white dark:bg-dark-card rounded-2xl border border-gray-100 dark:border-white/5">All cameras are disabled. Enable them to view live feeds.</div>`;
                    } else {
                        previewGrid.innerHTML = activeCams.map(cam => `
                            <div class="bg-white dark:bg-dark-card rounded-2xl p-4 shadow-sm border border-gray-100 dark:border-white/5 flex flex-col gap-4 overflow-hidden relative group">
                                <div class="flex justify-between items-start">
                                    <div class="truncate pr-4">
                                        <h3 class="font-bold text-text-primary dark:text-white truncate" title="${cam.label || cam.id}">${cam.label || cam.id}</h3>
                                        <p class="text-xs text-text-tertiary mt-0.5 font-mono truncate" title="${cam.id}">${cam.id}</p>
                                    </div>
                                    <div class="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-50 dark:bg-green-500/10 border border-green-200/50 dark:border-green-500/20 shrink-0">
                                        <div class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
                                        <span class="text-[10px] font-bold text-green-700 dark:text-green-400 tracking-wide uppercase">Live</span>
                                    </div>
                                </div>
                                <div class="w-full aspect-video bg-black rounded-xl overflow-hidden relative shadow-inner">
                                    <img src="/api/stream/${cam.id}" class="w-full h-full object-contain" 
                                         onerror="const el=this; setTimeout(() => { el.src = `/api/stream/${cam.id}?t=${Date.now()}`; }, 2500);">
                                </div>
                            </div>
                        `).join('');
                    }
                }
            }
        }

        async function toggleCam(id, enable) {
            await fetch(`/api/cameras/${id}`, {
                credentials: 'same-origin',
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: enable})
            });
            fetchCameras();
        }

        async function deleteCam(id) {
            if (!confirm('Remove this camera?')) return;
            await fetch(`/api/cameras/${id}`, {
                credentials: 'same-origin', method: 'DELETE', credentials: 'same-origin' });
            fetchCameras();
        }

        function showCamMsg(msg, isError=false) {
            const el = document.getElementById('cam-msg');
            el.textContent = msg;
            el.className = `text-xs mt-2 ${isError ? 'text-red-500' : 'text-green-600 dark:text-green-400'}`;
            el.classList.remove('hidden');
            // setTimeout(() => el.classList.add('hidden'), 5000);
        }

        async function testCamera() {
            const source = document.getElementById('cam-source').value.trim();
            if (!source) return showCamMsg('Please enter a source to test', true);
            
            showCamMsg('Testing connection... (this may take a few seconds)');
            try {
                const res = await fetch('/api/cameras/test', {
                    credentials: 'same-origin',
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({source: source, label: '', enabled: true})
                });
                const data = await res.json();
                if (data.success) {
                    showCamMsg('✓ Connection successful!');
                } else {
                    showCamMsg(`✗ Test failed: ${data.error}`, true);
                }
            } catch (e) {
                showCamMsg('✗ Test failed: Network error', true);
            }
        }

        
        // ── Remote Camera Status ────────────────────────────────────────────
        async function fetchRemoteCameras() {
            try {
                const res = await fetch('/api/camera/remote/status');
                const data = await res.json();
                const container = document.getElementById('remote-cam-list');
                if (!data.cameras || data.cameras.length === 0) {
                    container.innerHTML = '<p class="text-[11px] text-text-tertiary italic">No remote cameras connected</p>';
                    return;
                }
                container.innerHTML = data.cameras.map(cam => {
                    const statusColor = cam.status === 'LIVE' ? 'text-green-500' : cam.status === 'STALE' ? 'text-yellow-500' : 'text-red-400';
                    const statusDot   = cam.status === 'LIVE' ? 'bg-green-500 animate-pulse' : cam.status === 'STALE' ? 'bg-yellow-500' : 'bg-red-400';
                    return `<div class="flex items-center justify-between bg-gray-50 dark:bg-white/5 rounded-lg px-3 py-2">
                        <div class="flex items-center gap-2">
                            <span class="text-base">🌐</span>
                            <div>
                                <p class="text-[11px] font-semibold text-text-primary dark:text-white">${cam.camera_id}</p>
                                <p class="text-[9px] text-text-tertiary">${cam.fps} fps · ${cam.frames_received} frames · ${cam.last_seen_ago_s ? cam.last_seen_ago_s + 's ago' : 'waiting'}</p>
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full ${statusDot}"></span>
                            <span class="text-[10px] font-bold ${statusColor}">${cam.status}</span>
                        </div>
                    </div>`;
                }).join('');
            } catch (e) {
                console.error('fetchRemoteCameras error:', e);
            }
        }

        // Poll remote cameras every 5s
        setInterval(fetchRemoteCameras, 5000);
        fetchRemoteCameras();

        async function addCamera() {
            const id = document.getElementById('cam-id').value.trim();
            const label = document.getElementById('cam-label').value.trim();
            const source = document.getElementById('cam-source').value.trim();
            if (!source) return showCamMsg('Please enter a camera source', true);

            showCamMsg('Adding camera and verifying connection...');
            try {
                const payload = {source: source, label: label, enabled: true};
                if (id) payload.id = id;
                const res = await fetch('/api/cameras', {
                    credentials: 'same-origin',
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                if (data.success) {
                    document.getElementById('cam-id').value = '';
                    document.getElementById('cam-label').value = '';
                    document.getElementById('cam-source').value = '';
                    showCamMsg('✓ Camera added successfully');
                    fetchCameras();
                } else {
                    showCamMsg(`✗ Failed: ${data.error}`, true);
                }
            } catch (e) {
                showCamMsg('✗ Failed to add camera: Network error', true);
            }
        }

        // Initialize camera list and refresh periodically
        fetchCameras();
        setInterval(fetchCameras, 10000);

        // ── Theme Logic ───────────────────────────────────────────────
        (function () {
                const themeToggleBtn = document.getElementById('theme-toggle');
                const htmlRoot = document.documentElement;
                const iconLight = document.querySelector('.theme-icon-light');
                const iconDark = document.querySelector('.theme-icon-dark');

                function updateThemeUI() {
                    const isDark = htmlRoot.classList.contains('dark');
                    if (isDark) {
                        iconLight.classList.remove('hidden');
                        iconDark.classList.add('hidden');
                    } else {
                        iconLight.classList.add('hidden');
                        iconDark.classList.remove('hidden');
                    }
                }

                // Initialize state
                if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                    htmlRoot.classList.add('dark');
                } else {
                    htmlRoot.classList.remove('dark');
                }
                updateThemeUI();

                themeToggleBtn.addEventListener('click', () => {
                    if (htmlRoot.classList.contains('dark')) {
                        htmlRoot.classList.remove('dark');
                        localStorage.theme = 'light';
                    } else {
                        htmlRoot.classList.add('dark');
                        localStorage.theme = 'dark';
                    }
                    updateThemeUI();
                });
            })();
    