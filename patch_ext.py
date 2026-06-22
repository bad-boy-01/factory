with open('core/memory/extractor.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('"role": ""\\n', '"role": "",\\n      "current_outfit": ""\\n')
c = c.replace('"description": ""\\n', '"description": "",\\n      "visual_tags": ""\\n')

with open('core/memory/extractor.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("Patched extractor")
