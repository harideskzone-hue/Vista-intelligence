
        // ── Live Pipeline Stats ───────────────────────────────────────────────
        function runStorageCleanup() {
            if (!confirm("Are you sure you want to run cleanup? This will permanently delete old recognition evidence and video segments.")) return;
            
            fetch('/api/system/cleanup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ retention_days: 30 })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert(`Cleanup completed:\n- Evidence images deleted: ${data.deleted_evidence}\n- Video recordings deleted: ${data.deleted_recordings}`);
                    refreshStats();
                } else {
                    alert("Cleanup failed: " + data.error);
                }
            })
            .catch(err => alert("Error: " + err));
        }

        // Auto-poll /api/health every 8 seconds so stats update as the
        // capture → scoring → upload pipeline adds new faces and images.
        let lastNetBytes = { sent: null, recv: null, time: null };

        function formatBytes(bytes) {
            if (bytes === 0) return { val: '0', unit: 'B' };
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return { val: parseFloat((bytes / Math.pow(k, i)).toFixed(2)), unit: sizes[i] };
        }

        function refreshStats() {
            fetch('/api/health')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('stat-persons').textContent = data.person_count ?? '—';
                    document.getElementById('stat-images').textContent = data.vector_count ?? '—';
                    
                    if (data.storage_used_bytes !== undefined) {
                        const s = formatBytes(data.storage_used_bytes);
                        document.getElementById('stat-storage-value').textContent = s.val;
                        document.getElementById('stat-storage-unit').textContent = s.unit;
                        
                        if (data.database_bytes !== undefined) {
                            const db = formatBytes(data.database_bytes);
                            document.getElementById('stat-db-value').textContent = db.val + db.unit;
                            
                            const ref = formatBytes(data.reference_images_bytes);
                            document.getElementById('stat-ref-value').textContent = ref.val + ref.unit;
                            
                            const evid = formatBytes(data.evidence_bytes);
                            document.getElementById('stat-evid-value').textContent = evid.val + evid.unit;
                            
                            const rec = formatBytes(data.recordings_bytes);
                            document.getElementById('stat-rec-value').textContent = rec.val + rec.unit;
                        }
                    }
                    
                    const now = performance.now() / 1000;
                    if (data.network_bytes_sent !== undefined && data.network_bytes_recv !== undefined) {
                        if (lastNetBytes.sent !== null) {
                            const dt = now - lastNetBytes.time;
                            if (dt > 0) {
                                const dSent = data.network_bytes_sent - lastNetBytes.sent;
                                const dRecv = data.network_bytes_recv - lastNetBytes.recv;
                                
                                const upMbps = (dSent * 8 / (dt * 1000000)).toFixed(2);
                                const downMbps = (dRecv * 8 / (dt * 1000000)).toFixed(2);
                                
                                const upEl = document.getElementById('stat-upload-speed');
                                const downEl = document.getElementById('stat-download-speed');
                                if (upEl) upEl.textContent = upMbps;
                                if (downEl) downEl.textContent = downMbps;
                            }
                        }
                        lastNetBytes = { sent: data.network_bytes_sent, recv: data.network_bytes_recv, time: now };
                    }
                })
                .catch(() => {});
        }
        refreshStats();                      // immediate first load
        setInterval(refreshStats, 8000);     // then every 8 s

        function showRecognitionToast(event) {
            const container = document.getElementById('toast-container');
            const name = event.matched_identity_name || event.matched_identity_id || 'Unknown';
            const timeStr = new Date(event.timestamp * 1000).toLocaleTimeString();
            
            const toast = document.createElement('div');
            toast.className = 'toast-enter pointer-events-auto bg-white dark:bg-dark-card border border-primary/30 shadow-[0_8px_30px_rgb(0,0,0,0.12)] rounded-xl p-4 flex items-center gap-4 w-80 relative overflow-hidden';
            
            const imgHtml = event.image 
                ? `<img src="data:image/jpeg;base64,${event.image}" class="h-14 w-14 object-cover rounded-lg shadow-sm border border-gray-100 dark:border-white/10">` 
                : `<div class="h-14 w-14 bg-gray-100 dark:bg-white/5 rounded-lg flex items-center justify-center text-xs text-gray-400">No Img</div>`;

            toast.innerHTML = `
                <div class="absolute left-0 top-0 bottom-0 w-1 bg-green-500"></div>
                ${imgHtml}
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-bold text-text-primary dark:text-white truncate">${name} is here!</p>
                    <p class="text-[11px] text-text-secondary mt-1 truncate font-mono">${event.camera_id}</p>
                    <p class="text-[10px] text-text-tertiary mt-0.5">${timeStr}</p>
                </div>
                <button class="absolute top-2 right-2 text-gray-400 hover:text-gray-600 dark:hover:text-white transition-colors" onclick="this.closest('.toast-enter').remove()">
                    <span class="material-symbols-outlined text-[16px]">close</span>
                </button>
            `;
            
            container.appendChild(toast);
            
            // Auto dismiss after 7 seconds
            setTimeout(() => {
                toast.classList.remove('toast-enter');
                toast.classList.add('toast-exit');
                setTimeout(() => toast.remove(), 400);
            }, 7000);
        }

        let lastAlertTimestamp = 0;
        function refreshLiveEvents() {
            fetch('/api/events/live?n=15')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('live-events-tbody');
                    const section = document.getElementById('recent-shots-section');
                    
                    if (!data.events || data.events.length === 0) {
                        section.classList.add('hidden');
                        return;
                    }
                    
                    // Trigger alert for new matches
                    const newMatches = data.events.filter(ev => ev.match_status === 'MATCH' && ev.timestamp > lastAlertTimestamp && (ev.temporal_state === 'CONFIRMED' || ev.temporal_state === 'LOCKED'));
                    if (newMatches.length > 0) {
                        lastAlertTimestamp = Math.max(...newMatches.map(ev => ev.timestamp));
                        // Show toast for each new match (newest first in the events array, but we want to show oldest first to stack properly)
                        newMatches.reverse().forEach(match => showRecognitionToast(match));
                    }
                    
                    section.classList.remove('hidden');
                    tbody.innerHTML = data.events.map(ev => {
                        let date = new Date(ev.timestamp * 1000);
                        let timeStr = date.toLocaleTimeString();
                        
                        let statusColor = 'text-gray-500';
                        if (ev.match_status === 'MATCH') statusColor = 'text-green-500 font-bold';
                        else if (ev.match_status === 'UNKNOWN') statusColor = 'text-red-500 font-bold';
                        else if (ev.match_status === 'AMBIGUOUS') statusColor = 'text-orange-500 font-bold';
                        
                        let distStr = ev.embedding_distance != null ? ev.embedding_distance.toFixed(3) : '—';
                        
                        let imgHtml = ev.image ? 
                            `<img src="data:image/jpeg;base64,${ev.image}" class="h-10 w-10 object-cover rounded shadow-sm border border-gray-200">` : 
                            `<span class="text-xs text-gray-400">—</span>`;
                            
                        let stateColor = 'text-gray-500';
                        if (ev.temporal_state === 'CONFIRMED') stateColor = 'text-blue-500 font-bold';
                        else if (ev.temporal_state === 'LOCKED') stateColor = 'text-green-600 font-bold';
                        else if (ev.temporal_state === 'CANDIDATE') stateColor = 'text-orange-400 font-bold';
                            
                        return `
                            <tr>
                                <td class="py-3 pr-4 whitespace-nowrap text-text-secondary">${timeStr}</td>
                                <td class="py-3 pr-4 whitespace-nowrap text-text-primary font-mono text-xs">${ev.camera_id}</td>
                                <td class="py-3 pr-4 whitespace-nowrap ${stateColor} text-xs tracking-wider">${ev.temporal_state || '—'}</td>
                                <td class="py-3 pr-4 whitespace-nowrap ${statusColor}">${ev.match_status || '—'}</td>
                                <td class="py-3 pr-4 whitespace-nowrap font-medium">${ev.matched_identity_name || ev.matched_identity_id || '—'}</td>
                                <td class="py-3 pr-4 whitespace-nowrap text-text-secondary font-mono text-xs">${distStr}</td>
                                <td class="py-3 pr-4 whitespace-nowrap text-text-secondary text-xs">${ev.action || '—'}</td>
                                <td class="py-3 whitespace-nowrap">${imgHtml}</td>
                            </tr>
                        `;
                    }).join('');
                })
                .catch(() => {});
        }
        refreshLiveEvents();
        setInterval(refreshLiveEvents, 2000); // Poll every 2 seconds
    