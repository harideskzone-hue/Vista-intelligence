import os

path = "face_api/templates/preview.html"
with open(path, "r") as f:
    content = f.read()

# Fix refreshLiveEvents
old_1 = """                    if (!data.events || data.events.length === 0) {
                        section.classList.add('hidden');
                        return;
                    }"""
new_1 = """                    if (!data.events || data.events.length === 0) {
                        if (section) section.classList.add('hidden');
                        return;
                    }"""
content = content.replace(old_1, new_1)

old_2 = """                    section.classList.remove('hidden');
                    tbody.innerHTML = data.events.map(ev => {"""
new_2 = """                    if (section) section.classList.remove('hidden');
                    if (tbody) {
                        tbody.innerHTML = data.events.map(ev => {"""
content = content.replace(old_2, new_2)

old_3 = """                        `;
                    }).join('');
                })
                .catch(() => {});"""
new_3 = """                        `;
                        }).join('');
                    }
                })
                .catch(() => {});"""
content = content.replace(old_3, new_3)

# Fix updateCameraUI
old_4 = """            if (active) {
                btn.textContent = "Stop Camera";
                btn.classList.add('bg-red-500', 'text-white', 'hover:bg-red-600');
                btn.classList.remove('bg-primary', 'dark:bg-white', 'dark:text-primary');
                ind.innerHTML = `"""
new_4 = """            if (active) {
                if (btn) {
                    btn.textContent = "Stop Camera";
                    btn.classList.add('bg-red-500', 'text-white', 'hover:bg-red-600');
                    btn.classList.remove('bg-primary', 'dark:bg-white', 'dark:text-primary');
                }
                if (ind) ind.innerHTML = `"""
content = content.replace(old_4, new_4)

old_5 = """            } else {
                btn.textContent = "Start Camera";
                btn.classList.remove('bg-red-500', 'text-white', 'hover:bg-red-600');
                btn.classList.add('bg-primary', 'dark:bg-white', 'dark:text-primary');
                ind.innerHTML = `"""
new_5 = """            } else {
                if (btn) {
                    btn.textContent = "Start Camera";
                    btn.classList.remove('bg-red-500', 'text-white', 'hover:bg-red-600');
                    btn.classList.add('bg-primary', 'dark:bg-white', 'dark:text-primary');
                }
                if (ind) ind.innerHTML = `"""
content = content.replace(old_5, new_5)

old_6 = """            if (pending_jobs > 0) {
                pipeInd.innerHTML = `"""
new_6 = """            if (pipeInd) {
                if (pending_jobs > 0) {
                    pipeInd.innerHTML = `"""
content = content.replace(old_6, new_6)

old_7 = """                    <span class="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Pipeline Idle</span>
                `;
            }
        }"""
new_7 = """                    <span class="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Pipeline Idle</span>
                `;
                }
            }
        }"""
content = content.replace(old_7, new_7)

# Fix missing closing backticks or syntax issues just in case
with open(path, "w") as f:
    f.write(content)

print("Applied fixes to preview.html")
