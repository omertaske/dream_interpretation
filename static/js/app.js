// Rüyalar Rehberi - Frontend JS
const $ = (sel) => document.querySelector(sel);

const form = $('#form');
const textEl = $('#text');
const lengthEl = $('#length');
const topkEl = $('#topk');
const loadingEl = $('#loading');
const errorEl = $('#error');
const resultEl = $('#result');
const rSummary = $('#r-summary');
const rItems = $('#r-items');
const rBody = $('#r-body');
const rTone = $('#r-tone');
const rRecs = $('#r-recs');
const symSearch = $('#sym-search');
const symSuggest = $('#sym-suggest');
const historyEl = $('#history');

const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
function clearResult() {
  rSummary.textContent = '';
  rItems.innerHTML = '';
  rBody.textContent = '';
  rTone.textContent = '';
  rRecs.innerHTML = '';
}

$('#btn-sample').addEventListener('click', () => {
  textEl.value = 'Rüyamda yılan gördüm, deniz kenarında yürüyordum ve altın buldum. Sonra karanlık bulutlar dağıldı ve içimde bir ferahlık hissettim.';
});
$('#btn-clear').addEventListener('click', () => {
  textEl.value = '';
  clearResult();
  hide(resultEl);
  hide(errorEl);
});
$('#btn-copy').addEventListener('click', async () => {
  const txt = composePlainText();
  try { await navigator.clipboard.writeText(txt); } catch {}
});
function composePlainText() {
  let out = '';
  out += 'Özet\n' + (rSummary.textContent || '') + '\n\n';
  out += 'Semboller\n';
  [...rItems.querySelectorAll('li')].forEach(li => { out += '- ' + li.textContent + '\n'; });
  out += '\nMuhtemel Yorum\n' + (rBody.textContent || '') + '\n\n';
  out += 'Genel Ton\n' + (rTone.textContent || '') + '\n\n';
  out += 'Öneriler\n';
  [...rRecs.querySelectorAll('li')].forEach(li => { out += '- ' + li.textContent + '\n'; });
  return out;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hide(errorEl);
  hide(resultEl);
  show(loadingEl);
  clearResult();

  const payload = {
    text: textEl.value.trim(),
    length: lengthEl.value,
    top_k: Number(topkEl.value) || 8,
  };

  if (!payload.text) {
    hide(loadingEl);
    errorEl.textContent = 'Lütfen rüyanızı yazın.';
    show(errorEl);
    return;
  }

  try {
    const res = await fetch('/api/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    hide(loadingEl);

    if (!res.ok || !data.ok) {
      errorEl.textContent = (data && data.error) || 'Bir sorun oluştu.';
      show(errorEl);
      return;
    }

    const out = data.result;
    rSummary.textContent = out.summary || '';

    (out.items || []).forEach((it) => {
      const li = document.createElement('li');
      const hintClass = it.hint === 'olumlu' ? 'positive' : it.hint === 'olumsuz' ? 'negative' : '';
      li.innerHTML = `<strong>${it.word}</strong> <span class="hint ${hintClass}">${it.hint}</span><br/>${(it.meaning || '').replace(/\n/g, '<br/>')}`;
      rItems.appendChild(li);
    });

    rBody.textContent = out.body || '';
    rTone.textContent = out.tone || '';
    (out.recommendations || []).forEach((t) => {
      const li = document.createElement('li');
      li.textContent = t;
      rRecs.appendChild(li);
    });

    show(resultEl);
    saveHistory(payload, out);
    renderHistory();
    // Yorumla'ya tıklayınca sonuç alanına yumuşak kaydır
    setTimeout(() => {
      resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
  } catch (err) {
    hide(loadingEl);
    errorEl.textContent = 'Ağ hatası: ' + (err && err.message ? err.message : err);
    show(errorEl);
  }
});
// Sembol arama (autocomplete)
let symTimer = 0;
symSearch.addEventListener('input', () => {
  clearTimeout(symTimer);
  const q = symSearch.value.trim();
  if (!q) { symSuggest.innerHTML=''; hide(symSuggest); return; }
  symTimer = setTimeout(async () => {
    try {
      const res = await fetch('/api/symbols?q=' + encodeURIComponent(q));
      const data = await res.json();
      if (!data.ok) return;
      symSuggest.innerHTML = '';
      (data.items || []).forEach(w => {
        const li = document.createElement('li');
        li.textContent = w;
        li.addEventListener('click', () => {
          insertAtCursor(textEl, w + ' ');
          symSuggest.innerHTML=''; hide(symSuggest);
          textEl.focus();
        });
        symSuggest.appendChild(li);
      });
      if ((data.items || []).length) show(symSuggest); else hide(symSuggest);
    } catch {}
  }, 200);
});
function insertAtCursor(el, text) {
  const start = el.selectionStart; const end = el.selectionEnd;
  const val = el.value;
  el.value = val.slice(0, start) + text + val.slice(end);
  el.selectionStart = el.selectionEnd = start + text.length;
}
// Geçmiş (localStorage)
const KEY_HISTORY = 'rr-history-v1';
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(KEY_HISTORY) || '[]'); } catch { return []; }
}
function saveHistory(input, result) {
  const item = { ts: Date.now(), input, result };
  const arr = loadHistory();
  arr.unshift(item);
  const trimmed = arr.slice(0, 5);
  localStorage.setItem(KEY_HISTORY, JSON.stringify(trimmed));
}
function renderHistory() {
  const arr = loadHistory();
  historyEl.innerHTML = '';
  arr.forEach((h, i) => {
    const li = document.createElement('li');
    const txt = (h.input && h.input.text) ? h.input.text.slice(0, 80) + (h.input.text.length>80?'…':'') : '';
    li.textContent = `#${i+1} • ${new Date(h.ts).toLocaleString()} • ${txt}`;
    li.addEventListener('click', () => {
      textEl.value = h.input.text || '';
      lengthEl.value = h.input.length || 'long';
      topkEl.value = h.input.top_k || 8;
      // Sonucu doğrudan render et
      clearResult();
      const out = h.result || {};
      rSummary.textContent = out.summary || '';
      (out.items || []).forEach((it) => {
        const li2 = document.createElement('li');
        const hintClass = it.hint === 'olumlu' ? 'positive' : it.hint === 'olumsuz' ? 'negative' : '';
        li2.innerHTML = `<strong>${it.word}</strong> <span class="hint ${hintClass}">${it.hint}</span><br/>${(it.meaning || '').replace(/\n/g, '<br/>')}`;
        rItems.appendChild(li2);
      });
      rBody.textContent = out.body || '';
      rTone.textContent = out.tone || '';
      (out.recommendations || []).forEach((t) => {
        const li3 = document.createElement('li');
        li3.textContent = t;
        rRecs.appendChild(li3);
      });
      show(resultEl);
    });
    historyEl.appendChild(li);
  });
}
renderHistory();
