import os

path = "face_api/templates/preview.html"
with open(path, "r") as f:
    content = f.read()

# Fix refreshStats
old_1 = """                    document.getElementById('stat-persons').textContent = data.person_count ?? '—';
                    document.getElementById('stat-images').textContent = data.vector_count ?? '—';"""
new_1 = """                    const statPersons = document.getElementById('stat-persons');
                    const statImages = document.getElementById('stat-images');
                    if (statPersons) statPersons.textContent = data.person_count ?? '—';
                    if (statImages) statImages.textContent = data.vector_count ?? '—';"""
content = content.replace(old_1, new_1)

with open(path, "w") as f:
    f.write(content)

print("Applied fixes to refreshStats in preview.html")
