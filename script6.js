
      /* Fetch a CSRF token from the server (session must be valid) */
      async function getCsrfToken() {
        try {
          const r = await fetch('/api/csrf', { credentials: 'same-origin' });
          if (!r.ok) return '';
          const d = await r.json();
          return d.token || '';
        } catch { return ''; }
      }

      /* ⏻  Shutdown: confirm → POST /api/shutdown */
      async function confirmShutdown() {
        if (!confirm('⏻  Shut down SIH26187?\n\nThis will stop the camera, AI pipeline, and close the server.')) return;
        const overlay = document.getElementById('shutdown-overlay');
        const overlayText = overlay.querySelector('p');
        overlay.style.display = 'flex';
        
        try {
          const csrf = await getCsrfToken();
          const fd = new FormData();
          fd.append('csrf_token', csrf);
          await fetch('/api/shutdown', { method: 'POST', body: fd, credentials: 'same-origin' });
        } catch { /* server going down — expected */ }

        // Attempt to close the window automatically after a short delay
        setTimeout(() => {
          overlayText.innerText = "System has been shut down successfully.\nThis tab will now close (or you can close it manually).";
          window.close();
        }, 1500);
      }

      /* 🚪 Logout: POST /auth/logout → redirect to /login */
      async function doLogout() {
        try {
          const csrf = await getCsrfToken();
          const fd = new FormData();
          fd.append('csrf_token', csrf);
          const r = await fetch('/auth/logout', { method: 'POST', body: fd, credentials: 'same-origin' });
          const d = await r.json().catch(() => ({}));
          window.location.href = d.redirect || '/login';
        } catch {
          window.location.href = '/login';
        }
      }
    