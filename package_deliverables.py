import shutil
import os
import zipfile

os.makedirs('deliverables', exist_ok=True)
shutil.copy2('figures/architecture.png', 'deliverables/architecture.png')
shutil.copy2('docs/literature_table.md', 'deliverables/literature_table.md')
shutil.copy2('data/sample/sample_100.csv', 'deliverables/sample_100.csv')
shutil.copy2('results/model_comparison.csv', 'deliverables/model_comparison.csv')

# Create deliverables.zip
with zipfile.ZipFile('deliverables.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk('deliverables'):
        for f in sorted(files):
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, '.')
            z.write(fp, arcname)
            print(f'Added to zip: {arcname}')

print('deliverables.zip created successfully!')
print(f'Zip size: {os.path.getsize("deliverables.zip"):,} bytes')
