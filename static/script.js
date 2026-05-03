const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');
const uploadCard  = document.getElementById('uploadCard');
const previewCard = document.getElementById('previewCard');
const previewImg  = document.getElementById('previewImg');
const btnChange   = document.getElementById('btnChange');
const btnPredict  = document.getElementById('btnPredict');
const resultBox   = document.getElementById('resultBox');
const loading     = document.getElementById('loading');
const idleMsg     = document.getElementById('idleMsg');

let selectedFile = null;

// ── File selection via input ──────────────────────────────────────────────────
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) loadFile(e.target.files[0]);
});

// ── Drag & Drop ───────────────────────────────────────────────────────────────
uploadCard.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadCard.classList.add('dragover');
});

uploadCard.addEventListener('dragleave', () => {
  uploadCard.classList.remove('dragover');
});

uploadCard.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadCard.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

// ── Load and preview file ─────────────────────────────────────────────────────
function loadFile(file) {
  const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
  if (!allowed.includes(file.type)) {
    alert('Format non supporté. Utilisez JPG ou PNG.');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadCard.style.display = 'none';
    previewCard.style.display = 'flex';
    resultBox.style.display   = 'none';
    idleMsg.style.display     = 'block';
    loading.style.display     = 'none';
    btnPredict.disabled       = false;
  };
  reader.readAsDataURL(file);
}

// ── Change image ──────────────────────────────────────────────────────────────
btnChange.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  uploadCard.style.display  = 'block';
  previewCard.style.display = 'none';
  resultBox.style.display   = 'none';
});

// ── Predict ───────────────────────────────────────────────────────────────────
btnPredict.addEventListener('click', async () => {
  if (!selectedFile) return;

  idleMsg.style.display   = 'none';
  resultBox.style.display = 'none';
  loading.style.display   = 'flex';
  btnPredict.disabled     = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    loading.style.display   = 'none';
    resultBox.style.display = 'flex';
    btnPredict.disabled     = false;

    if (data.error) {
      showError(data.error);
      return;
    }

    showResult(data);

  } catch (err) {
    loading.style.display = 'none';
    btnPredict.disabled   = false;
    showError('Erreur de connexion au serveur.');
  }
});

// ── Display result ────────────────────────────────────────────────────────────
function showResult(data) {
  const icon        = document.getElementById('resultIcon');
  const label       = document.getElementById('resultLabel');
  const bar         = document.getElementById('confidenceBar');
  const confText    = document.getElementById('confidenceText');
  const probText    = document.getElementById('probabilityText');

  icon.textContent  = data.status === 'positive' ? '⚠️' : '✅';
  label.textContent = data.result;
  label.className   = 'result-label ' + data.status;

  bar.className     = 'confidence-bar ' + data.status;
  bar.style.width   = '0%';
  setTimeout(() => { bar.style.width = data.confidence + '%'; }, 50);

  confText.textContent = `Confiance : ${data.confidence}%`;
  probText.textContent = `Probabilité brute : ${data.probability}`;
}

function showError(msg) {
  const label = document.getElementById('resultLabel');
  document.getElementById('resultIcon').textContent = '❌';
  label.textContent = msg;
  label.className   = 'result-label positive';
  document.getElementById('confidenceBar').style.width = '0%';
  document.getElementById('confidenceText').textContent = '';
  document.getElementById('probabilityText').textContent = '';
}
