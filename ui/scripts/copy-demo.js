/**
 * Copy built UI (dist) to examples/3k-platform-demo so the demo serves the real platform UI.
 * Run from ui directory: node scripts/copy-demo.js
 */
const fs = require('fs');
const path = require('path');

const distDir = path.join(__dirname, '..', 'dist');
const demoDir = path.join(__dirname, '..', '..', 'examples', '3k-platform-demo');

if (!fs.existsSync(distDir)) {
  console.error('Run "npm run build:demo" first. dist/ not found.');
  process.exit(1);
}

if (!fs.existsSync(demoDir)) {
  fs.mkdirSync(demoDir, { recursive: true });
}

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    for (const name of fs.readdirSync(src)) {
      copyRecursive(path.join(src, name), path.join(dest, name));
    }
  } else {
    fs.copyFileSync(src, dest);
  }
}

// Clear demo dir except preserved docs
const preserveFiles = ['README.md', '使用说明.md', '安装Node最简步骤.md'];
const preserved = {};
for (const name of preserveFiles) {
  const fullPath = path.join(demoDir, name);
  if (fs.existsSync(fullPath)) {
    preserved[name] = fs.readFileSync(fullPath, 'utf8');
  }
}
for (const name of fs.readdirSync(demoDir)) {
  const p = path.join(demoDir, name);
  if (preserveFiles.includes(name)) continue;
  fs.rmSync(p, { recursive: true, force: true });
}

// Copy dist contents into demo dir
for (const name of fs.readdirSync(distDir)) {
  copyRecursive(path.join(distDir, name), path.join(demoDir, name));
}

for (const [name, content] of Object.entries(preserved)) {
  fs.writeFileSync(path.join(demoDir, name), content);
}

console.log('Demo updated at examples/3k-platform-demo');
