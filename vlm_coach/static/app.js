const app = document.querySelector('#app');
const runtime = document.querySelector('#runtime');
const toast = document.querySelector('#toast');
let pollTimer;

const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const api = async (path, options = {}) => {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
};
const notify = message => { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2600); };

async function checkRuntime() {
  try {
    const result = await api('/api/health');
    const provider = result.provider || result.ollama;
    if (provider.available) {
      const models = provider.models.length ? provider.models.join(', ') : 'no models installed';
      runtime.className = 'runtime good'; runtime.textContent = `${(provider.provider || 'local VLM').toUpperCase()} · ${models}`;
    } else {
      runtime.className = 'runtime bad'; runtime.textContent = 'Ollama is offline';
    }
  } catch { runtime.className = 'runtime bad'; runtime.textContent = 'Backend unavailable'; }
}

function route(matchId) {
  clearInterval(pollTimer);
  history.pushState({}, '', matchId ? `/?match=${matchId}` : '/');
  matchId ? showMatch(matchId) : showHome();
}

async function showHome() {
  clearInterval(pollTimer);
  const { matches } = await api('/api/matches');
  app.innerHTML = `
    <section class="hero">
      <div><span class="eyebrow">A match review, not a fake referee</span><h1>Turn match video into a coaching story.</h1><p class="lede">Local Qwen reviews short visual storyboards, links every observation back to a rally, and gives you a practical training plan.</p></div>
      <aside class="promise"><b>The promise</b><p>One hour of padel becomes a ten-minute review of patterns, highlights and what to practise next.</p></aside>
    </section>
    <div class="grid">
      <form id="upload" class="card upload-card">
        <h2>Review a new match</h2>
        <div class="fields">
          <div class="field wide"><label>Recording</label><input name="file" type="file" accept="video/*" required><div class="help">Fixed landscape view with the whole court visible works best.</div></div>
          <div class="field wide"><label>Match name</label><input name="name" value="Friday padel" required></div>
          <div><label>Team A</label><input name="team_a" value="Team A"></div>
          <div><label>Team B</label><input name="team_b" value="Team B"></div>
          <div class="field wide"><label>Local vision model</label><select name="model"><option value="qwen3.5:2b">Qwen3.5 2B — recommended</option><option value="qwen3.5:0.8b">Qwen3.5 0.8B — fastest</option><option value="qwen3.5:4b">Qwen3.5 4B — tighter on 8 GB</option></select></div>
        </div>
        <button class="button accent" type="submit">Upload recording</button>
      </form>
      <div class="card"><span class="eyebrow">How it works</span><h2 style="margin-top:8px">Sparse, local and honest</h2><p>Motion finds candidate play windows. Qwen receives eight chronological images per window—not the full video—and returns strict JSON.</p><p>It focuses on positioning, transitions and coaching patterns. It marks uncertain winners and line calls as unknown.</p><p class="help">Your recording and model stay on this Mac.</p></div>
    </div>
    <div class="section-head"><h2>Previous reviews</h2><span>${matches.length} match${matches.length === 1 ? '' : 'es'}</span></div>
    <div class="match-list">${matches.length ? matches.map(match => `<div class="match-row" data-match="${match.id}"><div><b>${esc(match.name)}</b><small>${esc(match.team_a)} vs ${esc(match.team_b)} · ${esc(match.model)}</small></div><span class="pill ${esc(match.status)}">${esc(match.status)}</span></div>`).join('') : '<div class="empty">Your reviewed matches will appear here.</div>'}</div>`;
  document.querySelector('#upload').addEventListener('submit', uploadMatch);
  document.querySelectorAll('[data-match]').forEach(row => row.addEventListener('click', () => route(row.dataset.match)));
}

async function uploadMatch(event) {
  event.preventDefault(); const button = event.currentTarget.querySelector('button');
  button.disabled = true; button.textContent = 'Uploading…';
  try { const match = await api('/api/matches', { method: 'POST', body: new FormData(event.currentTarget) }); route(match.id); }
  catch (error) { notify(error.message); button.disabled = false; button.textContent = 'Upload recording'; }
}

async function showMatch(id) {
  clearInterval(pollTimer);
  try {
    const match = await api(`/api/matches/${id}`);
    if (match.status === 'analyzing') return renderProgress(match);
    renderMatch(match);
  } catch (error) { app.innerHTML = `<div class="error-box">${esc(error.message)}</div>`; }
}

function header(match) {
  return `<div class="detail-head"><div><button class="button ghost" data-home>← All matches</button><span class="eyebrow" style="display:block;margin-top:24px">${esc(match.model)} · local analysis</span><h1>${esc(match.name)}</h1><div class="teams"><b>${esc(match.team_a)}</b> versus <b>${esc(match.team_b)}</b></div></div><button class="button ghost danger" id="delete">Delete match</button></div>`;
}

function renderMatch(match) {
  if (match.status === 'uploaded' || match.status === 'error') {
    app.innerHTML = `${header(match)}<div class="card progress-card"><span class="eyebrow">Ready for local Qwen</span><h2>${match.status === 'error' ? 'Analysis needs attention' : 'Create the match review'}</h2>${match.error ? `<div class="error-box">${esc(match.error)}</div>` : ''}<p>Qwen will receive eight compressed frames from each candidate play window.</p><button id="analyze" class="button accent">Analyze recording</button></div>`;
    bindCommon(match); document.querySelector('#analyze').addEventListener('click', () => startAnalysis(match.id)); return;
  }
  const story = match.story;
  app.innerHTML = `${header(match)}
    <div class="video-shell"><video controls preload="metadata" src="${match.video_url}"></video></div>
    ${story ? renderStory(story, match) : ''}
    <div class="section-head"><h2>Evidence timeline</h2><span>${match.rallies.length} candidate segments</span></div>
    <div class="rally-list">${match.rallies.map(rally => renderRally(rally, match)).join('')}</div>`;
  bindCommon(match);
  document.querySelectorAll('.storyboard img').forEach(image => image.addEventListener('click', () => { const video = document.querySelector('video'); video.currentTime = Number(image.dataset.time); video.play(); window.scrollTo({top: 100, behavior: 'smooth'}); }));
  document.querySelectorAll('.review').forEach(form => form.addEventListener('submit', saveReview));
  const regenerate = document.querySelector('#regenerate'); if (regenerate) regenerate.addEventListener('click', () => regenerateStory(match.id));
}

function renderProgress(match) {
  const live = match.rallies || [];
  app.innerHTML = `${header(match)}<div class="card progress-card"><span class="eyebrow">Local analysis</span><h2>${esc(match.stage)}</h2><p>Processing stays on this Mac. New observations appear below as Qwen finishes each segment.</p><div class="progress-track"><span style="width:${match.progress}%"></span></div><b>${match.progress}%</b></div>${live.length ? `<div class="section-head"><h2>Live coaching feed</h2><span>${live.length} interpreted</span></div><div class="rally-list">${live.slice().reverse().map(rally => `<article class="card"><span class="eyebrow">Segment ${rally.id} · ${clock(rally.start)}–${clock(rally.end)}</span><p class="rally-summary" style="margin-top:8px">${esc(rally.analysis.summary)}</p><div class="chips"><span class="chip">${Math.round(rally.analysis.confidence * 100)}% confidence</span><span class="chip">${Math.round(rally.analysis.highlight_score * 100)}% highlight</span></div></article>`).join('')}</div>` : ''}`;
  bindCommon(match); pollTimer = setInterval(() => showMatch(match.id), 1200);
}

function renderStory(story, match) {
  const performance = match.performance ? `${match.performance.storyboards} storyboards · ${match.performance.images_sent} images · ${match.performance.vlm_seconds}s local VLM · ${match.performance.realtime_factor}× video length` : '';
  return `<section class="story"><span class="eyebrow">Match story</span><h2>${esc(story.headline)}</h2><p>${esc(story.overview)}</p>${performance ? `<div class="chips"><span class="chip" style="background:#33463b;color:#d8ff66">${esc(performance)}</span></div>` : ''}<div class="story-grid"><article><h3>${esc(match.team_a)}</h3><p>${esc(story.team_a_story)}</p></article><article><h3>${esc(match.team_b)}</h3><p>${esc(story.team_b_story)}</p></article></div>${story.training_priorities?.length ? `<div class="priorities">${story.training_priorities.map(item => `<div class="priority"><b>${esc(item.title)}</b><small>Rallies ${item.evidence_rallies.join(', ') || '—'}</small><p>${esc(item.reason)}</p><small>DRILL</small><div>${esc(item.suggested_drill)}</div></div>`).join('')}</div>` : ''}<div style="margin-top:22px"><button id="regenerate" class="button ghost" style="color:white;border-color:#526159">Rebuild story from my corrections</button></div></section>`;
}

function renderRally(rally, match) {
  const analysis = rally.analysis; const review = rally.review || {};
  return `<article class="card rally"><div class="rally-top"><div><b>Segment ${rally.id}</b><span class="rally-time"> · ${clock(rally.start)}–${clock(rally.end)} · Qwen ${rally.inference_seconds ?? '—'}s</span></div><span class="score">${Math.round(analysis.highlight_score * 100)}% highlight</span></div><div class="storyboard">${rally.storyboard.map(frame => `<img src="${frame.url}" loading="lazy" data-time="${frame.timestamp}" title="Jump to ${clock(frame.timestamp)}">`).join('')}</div><div class="rally-body"><p class="rally-summary">${esc(analysis.summary)}</p><div class="chips"><span class="chip">${esc(analysis.rally_quality)} quality</span><span class="chip">${Math.round(analysis.confidence * 100)}% interpretation confidence</span><span class="chip">likely winner: ${esc(analysis.ending.likely_winner)}</span><span class="chip">ending: ${esc(analysis.ending.type)}</span></div>${analysis.coaching_observations.map(item => `<div class="observation"><b>${esc(item.category.replace('_',' '))}:</b> ${esc(item.observation)} <small>(${Math.round(item.confidence * 100)}%)</small></div>`).join('')}${analysis.uncertainty.length ? `<p class="help">Uncertain: ${analysis.uncertainty.map(esc).join(' · ')}</p>` : ''}<form class="review" data-id="${rally.id}" data-match="${match.id}"><label>Who won?<select name="winner"><option value="unknown">Unknown</option><option value="team_a" ${review.winner==='team_a'?'selected':''}>${esc(match.team_a)}</option><option value="team_b" ${review.winner==='team_b'?'selected':''}>${esc(match.team_b)}</option></select></label><label>How?<select name="ending">${['unknown','winner','forced_error','unforced_error','net','out','double_bounce'].map(value => `<option ${review.ending===value?'selected':''} value="${value}">${value.replace('_',' ')}</option>`).join('')}</select></label><label>Coach note<input name="note" value="${esc(review.note || '')}" placeholder="What really happened?"></label><button class="button ghost" type="submit">Save</button></form></div></article>`;
}

async function startAnalysis(id) {
  try { await api(`/api/matches/${id}/analyze`, {method:'POST'}); showMatch(id); }
  catch (error) { notify(error.message); }
}
async function saveReview(event) {
  event.preventDefault(); const form = event.currentTarget; const payload = Object.fromEntries(new FormData(form));
  try { await api(`/api/matches/${form.dataset.match}/rallies/${form.dataset.id}`, {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); notify('Correction saved'); }
  catch (error) { notify(error.message); }
}
async function regenerateStory(id) {
  const button = document.querySelector('#regenerate'); button.disabled = true; button.textContent = 'Rebuilding…';
  try { await api(`/api/matches/${id}/story`, {method:'POST'}); await showMatch(id); notify('Story rebuilt from reviewed evidence'); }
  catch (error) { notify(error.message); button.disabled = false; button.textContent = 'Rebuild story from my corrections'; }
}
function bindCommon(match) {
  document.querySelectorAll('[data-home]').forEach(button => button.addEventListener('click', () => route()));
  document.querySelector('#delete')?.addEventListener('click', async () => { if (!confirm(`Delete ${match.name}?`)) return; try { await api(`/api/matches/${match.id}`, {method:'DELETE'}); route(); } catch (error) { notify(error.message); } });
}

document.querySelector('[data-home]').addEventListener('click', () => route());
window.addEventListener('popstate', () => { const id = new URLSearchParams(location.search).get('match'); id ? showMatch(id) : showHome(); });
checkRuntime(); const initial = new URLSearchParams(location.search).get('match'); initial ? showMatch(initial) : showHome();
