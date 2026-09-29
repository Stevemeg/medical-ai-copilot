
const API = window.location.origin;
let accessToken = '', roles = [];
let selectedPatient = null;
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = value => escapeHtml(value ?? 'Not supplied');
function table(columns, rows){
  if(!rows.length)return '<p class="muted">No records supplied.</p>';
  return `<table><thead><tr>${columns.map(c=>`<th>${escapeHtml(c[0])}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${text(c[1](r))}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function card(title, body){return `<div class="card"><h3>${escapeHtml(title)}</h3>${body}</div>`;}
function message(value, error=false){$('message').innerHTML=value?`<p class="${error?'error':'notice'}">${escapeHtml(value)}</p>`:'';}
async function request(path, options={}){
  const response=await fetch(API+path,{...options,headers:{'Content-Type':'application/json',...(accessToken?{'Authorization':`Bearer ${accessToken}`}:{ }),...(options.headers||{})}});
  const body=await response.json();
  if(!response.ok)throw new Error(body.detail?.message||`Request failed (${response.status})`);
  return body;
}
function show(tab){
  document.querySelectorAll('main > section').forEach(x=>x.classList.toggle('hidden',x.id!==tab));
  document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x.dataset.tab===tab));
}
document.querySelectorAll('nav button').forEach(button=>button.addEventListener('click',()=>show(button.dataset.tab)));
async function demos(){const rows=await request('/v1/demo-patients');$('demo').innerHTML=rows.map(x=>`<option value="${escapeHtml(x.key)}">${escapeHtml(x.label)}</option>`).join('');}
async function patients(){
  const rows=await request('/v1/patients');
  $('patient-list').innerHTML=rows.map(x=>`<option value="${escapeHtml(x.patient_id)}">${escapeHtml(x.synthetic_label)} · ${escapeHtml(x.source_patient_id)}</option>`).join('');
  if(rows.length){selectedPatient=rows.some(x=>x.patient_id===selectedPatient)?selectedPatient:rows[0].patient_id;$('patient-list').value=selectedPatient;await patientDetail();}
  else{$('patient-detail').innerHTML='<p class="muted">Load a bundled synthetic patient to begin.</p>';}
}
async function patientDetail(){
  if(!selectedPatient)return;
  const data=await request(`/v1/patients/${encodeURIComponent(selectedPatient)}`), c=data.context;
  const active=c.conditions.filter(x=>x.clinical_status==='active');
  const meds=c.medications.filter(x=>x.status==='active');
  const allergies=c.allergies.filter(x=>x.clinical_status!=='inactive'&&!['refuted','entered-in-error'].includes(x.verification_status));
  const observations=c.observations.filter(x=>!['entered-in-error','cancelled'].includes(x.status)).sort((a,b)=>(b.effective||'').localeCompare(a.effective||''));
  const value=o=>o.value?`${o.value.value} ${o.value.unit||''}`:o.components.map(x=>`${x.code.text||x.code.coding[0]?.display}: ${x.value.value} ${x.value.unit||''}`).join(', ');
  $('patient-detail').innerHTML=
    card(`${c.patient.synthetic_label} · ${c.patient.source_patient_id}`,`<p>Birth date: ${text(c.patient.birth_date)} · Gender as recorded: ${text(c.patient.gender)}</p><small>Snapshot ${escapeHtml(data.context_hash.slice(0,16))}…</small>`)+
    `<div class="grid"><div>${card('Conditions',table([['Condition',x=>x.code.text||x.code.coding[0]?.display],['Status',x=>x.clinical_status],['Onset',x=>x.onset]],active))}${card('Medications',table([['Medication',x=>x.code.text||x.code.coding[0]?.display],['Status',x=>x.status],['Authored',x=>x.authored_on]],meds))}${card('Allergies',table([['Allergy',x=>x.code.text||x.code.coding[0]?.display],['Status',x=>x.clinical_status],['Verification',x=>x.verification_status]],allergies))}</div><div>${card('Recent observations',table([['Observation',x=>x.code.text||x.code.coding[0]?.display],['Value',value],['Date',x=>x.effective]],observations.slice(0,8)))}${card('Encounters',table([['Type',x=>x.type?.text||'Encounter'],['Status',x=>x.status],['Start',x=>x.start]],c.encounters))}${card('Data availability',`<p>Supplied: ${text(data.data_availability.available_data_types.join(', ')||'none')}</p><p>Not supplied: ${text(data.data_availability.missing_data_types.join(', ')||'none')}</p>`)}</div></div>`+
    card('Timeline',table([['When',x=>x.timestamp||'Undated'],['Type',x=>x.event_type],['Event',x=>x.title]],data.timeline));
  await reviewHistory();
}
async function reviewHistory(){
  const rows=await request(`/v1/patients/${encodeURIComponent(selectedPatient)}/reviews`);
  $('review-history').innerHTML=rows.length?card('Saved reviews',rows.map(x=>`<p><button data-review="${escapeHtml(x.review_id)}">${escapeHtml(x.created_at)} · ${escapeHtml(x.status)}</button></p>`).join('')):'';
  document.querySelectorAll('[data-review]').forEach(button=>button.addEventListener('click',async()=>{try{renderReview(await request(`/v1/reviews/${encodeURIComponent(button.dataset.review)}`));}catch(e){message(e.message,true);}}));
  $('review-detail').innerHTML='';
}
async function renderReview(r){
  show('review');
  $('review-detail').innerHTML=card('Immutable review snapshot',`<p>Status: ${text(r.status)} | Snapshot ${escapeHtml(r.patient_context_hash.slice(0,16))}</p>
    <label for="review-date">Evaluation date</label> <input id="review-date" type="date" value="${new Date().toISOString().slice(0,10)}">
    <details><summary>Explicit record completeness assertion</summary><p>Use only when complete Encounter and Procedure history is known for this interval. FHIR fetch success alone does not establish coverage.</p><label><input id="coverage-confirm" type="checkbox">I assert complete Encounter and Procedure coverage</label><label for="coverage-start">Coverage start</label><input id="coverage-start" type="date"></details>
    ${r.status==='completed'?'':'<button id="evaluate-review" class="primary">Evaluate deterministic rules</button><button id="complete-review">Complete review</button>'}
    <div id="finding-list"></div><h3>Record information</h3><p>Supplied: ${text(r.data_availability.available_data_types.join(', '))}</p><p>Missing: ${text(r.data_availability.missing_data_types.join(', '))}</p>`);
  async function findings(){
    const rows=await request(`/v1/reviews/${r.review_id}/findings`);
    $('finding-list').innerHTML=rows.length?rows.map(f=>card(`${f.title} | ${f.status.replaceAll('_',' ')}`,`<p>${text(f.rationale)}</p><p>Missing information: ${text(f.missing_data.join(', ')||'none')}</p><p>Rule ${text(f.rule_id)} | version ${text(f.rule_version)} | evaluated ${text(f.evaluated_as_of)}</p>${f.evidence_refs.map(e=>card(`${e.publisher} | ${e.guideline_title}`,`<p>${text(e.version_id||'Authoritative recommendation snapshot')} | ${text(e.jurisdiction)} | ${text(e.lifecycle_status||e.verification_status)}</p><p>Recommendation ${text(e.recommendation_id)} | verified ${text(e.source_verified_on)}</p><a href="${escapeHtml(e.canonical_source_url)}" target="_blank" rel="noopener">Inspect authoritative evidence</a>`)).join('')}<label for="action-${f.finding_id}">Clinician disposition</label><select id="action-${f.finding_id}"><option value="accept">Accept for review</option><option value="dismiss">Dismiss</option><option value="already_addressed">Already addressed</option><option value="incorrect_evidence">Incorrect evidence</option><option value="not_clinically_relevant">Not clinically relevant</option></select><label for="note-${f.finding_id}">Note</label><input id="note-${f.finding_id}" maxlength="2000"><button data-action="${f.finding_id}">Record disposition</button><div id="actions-${f.finding_id}"></div>`)).join(''):'<p>No evaluation has been recorded.</p>';
    for(const f of rows){
      const actions=await request(`/v1/findings/${f.finding_id}/actions`);
      $(`actions-${f.finding_id}`).innerHTML=table([['Disposition',a=>a.action_type],['Note',a=>a.note],['Recorded',a=>a.created_at]],actions);
    }
    document.querySelectorAll('[data-action]').forEach(b=>b.onclick=async()=>{try{const id=b.dataset.action;await request(`/v1/findings/${id}/actions`,{method:'POST',body:JSON.stringify({action_type:$(`action-${id}`).value,note:$(`note-${id}`).value||null})});await findings();}catch(e){message(e.message,true);}});
  }
  $('evaluate-review')?.addEventListener('click',async()=>{try{const as_of=$('review-date').value;let record_coverage=null;if($('coverage-confirm').checked){if(!$('coverage-start').value)throw new Error('Enter the coverage start date.');record_coverage={start_date:$('coverage-start').value,end_date:as_of,complete_resource_types:['Encounter','Procedure']};}await request(`/v1/reviews/${r.review_id}/evaluate`,{method:'POST',body:JSON.stringify({as_of,record_coverage})});await findings();}catch(e){message(e.message,true);}});
  $('complete-review')?.addEventListener('click',async()=>{try{await renderReview(await request(`/v1/reviews/${r.review_id}/complete`,{method:'POST'}));}catch(e){message(e.message,true);}});
  await findings();
}
$('load-demo').onclick=async()=>{try{const result=await request(`/v1/demo-patients/${encodeURIComponent($('demo').value)}/load`,{method:'POST'});selectedPatient=result.patient_id;await patients();message('Synthetic patient loaded.');}catch(e){message(e.message,true);}};
$('import-file').onclick=async()=>{try{const file=$('bundle-file').files[0];if(!file)throw new Error('Select a JSON Bundle first.');const raw=JSON.parse(await file.text());const validation=await request('/v1/fhir/validate',{method:'POST',body:JSON.stringify(raw)});if(!validation.valid)throw new Error(validation.errors.map(x=>x.message).join('; '));const result=await request('/v1/patients/import',{method:'POST',body:JSON.stringify(raw)});selectedPatient=result.patient_id;await patients();message('Synthetic patient imported.');}catch(e){message(e.message,true);}};
$('patient-list').onchange=async e=>{selectedPatient=e.target.value;await patientDetail();};
$('refresh').onclick=()=>patients().catch(e=>message(e.message,true));
$('create-review').onclick=async()=>{try{if(!selectedPatient)throw new Error('Select a patient first.');const review=await request(`/v1/patients/${encodeURIComponent(selectedPatient)}/reviews`,{method:'POST'});await reviewHistory();renderReview(review);}catch(e){message(e.message,true);}};
function evidenceCard(e){return card(`${e.publisher} · ${e.canonical_title}`,`<p>${text(e.version_id)} · ${text(e.updated_at||e.published_at)} · ${text(e.jurisdiction)} · ${text(e.lifecycle_status)}</p><p>${e.recommendation_id?`Recommendation ${text(e.recommendation_id)} · `:''}Section ${text(e.section)} · pages ${text(e.page_start)}-${text(e.page_end)}</p><blockquote>${text(e.supporting_excerpt)}</blockquote>${e.canonical_source_url?`<a href="${escapeHtml(e.canonical_source_url)}" target="_blank" rel="noopener">Open source</a>`:''}`);}
$('ask').onclick=async()=>{try{const query=$('question').value.trim();if(!query)throw new Error('Enter a question.');const intent=$('intent').value;$('answer').textContent='Searching governed evidence…';const result=await request('/v1/evidence/query',{method:'POST',body:JSON.stringify({query,intent})});const byId=Object.fromEntries((result.evidence||[]).map(e=>[e.evidence_unit_id,e]));let html=intent==='historical'?'<p class="error">Historical / superseded evidence. Do not treat as current guidance.</p>':'';if(result.status==='abstained')html+='<p class="notice">The indexed evidence does not support a reliable answer.</p>';else if(result.status==='conflict')html+='<p class="error">Evidence differs. Review each source; no recommendation was selected.</p>';else html+=`<pre>${escapeHtml(result.answer_text)}</pre>`;for(const claim of result.claims||[]){html+=card(`Claim · ${claim.support_status}`,`<p>${text(claim.text)}</p>${claim.evidence_ids.map(id=>byId[id]?evidenceCard({...byId[id],supporting_excerpt:claim.verification_passages?.[id]||byId[id].supporting_excerpt}):'').join('')}`);}for(const conflict of result.conflicts||[]){html+=card('Evidence differs',`<p>${text(conflict.description)}</p>${conflict.evidence_ids.map(id=>byId[id]?evidenceCard(byId[id]):'').join('')}`);}$('answer').innerHTML=html;}catch(e){message(e.message,true);}};
async function sources(){const documents=await request('/v1/guidelines');const rows=documents.flatMap(d=>d.versions.map(v=>({...d,...v})));$('source-list').innerHTML=table([['Title',x=>x.canonical_title],['Publisher',x=>x.publisher],['Type',x=>x.source_type],['Lifecycle',x=>x.status==='superseded'||x.status==='historical'?'HISTORICAL - '+x.status:x.status],['Version',x=>x.version_id]],rows);}


async function updates(){
  const rows=await request('/v1/guideline-updates');
  const editor=roles.some(x=>['guideline_editor','clinical_admin'].includes(x));
  $('update-list').innerHTML=rows.length?rows.map(c=>card(`${c.document_id} | ${c.status}`,`<p>Current at discovery: ${text(c.base_version_id)} | Candidate: ${text(c.version_id)}</p><p>Detected ${text(c.detected_at)}</p><small>SHA-256 ${text(c.checksum)}</small><p>Added ${c.diff.added.length} | Removed ${c.diff.removed.length} | Modified ${c.diff.modified.length} | Unchanged ${c.diff.unchanged.length} | Ambiguous ${c.diff.ambiguous.length}</p>${c.diff.modified.map(d=>`<details><summary>Recommendation ${text(d.key)}</summary><pre>${text(d.text_diff)}</pre></details>`).join('')}<p>Affected rules: ${text(c.diff.affected_rules.map(r=>r.rule_id+' '+r.rule_version).join(', ')||'none identified')}</p>${editor?`<label for="decision-${c.candidate_id}">Review reason</label><input id="decision-${c.candidate_id}" minlength="5" maxlength="1000">${(c.status==='quarantined'?['prepare']:c.status==='review_required'?['approve','reject']:c.status==='approved'?['activate']:[]).map(a=>`<button data-candidate="${c.candidate_id}" data-decision="${a}">${a}</button>`).join('')}`:''}`)).join(''):'<p>No candidate updates. Registered sources require manual version review.</p>';
  document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=async()=>{try{await request(`/v1/guideline-updates/${b.dataset.candidate}/${b.dataset.decision}`,{method:'POST',body:JSON.stringify({note:$(`decision-${b.dataset.candidate}`).value})});await updates();await sources();}catch(e){message(e.message,true);}});
}
async function status(){const s=await request('/v1/system-status');$('system-status').innerHTML=card('Evaluation record',`<p>Last run: ${text(s.last_evaluation)} | Dataset ${text(s.dataset_version)} | ${s.evaluation_passed?'PASS':'Not passing or not available'}</p><p>Mode: ${text(s.mode)} | Knowledge provenance: ${text(s.knowledge_provenance)}</p>`)+card('Retrieval',table([['Metric',x=>x[0]],['Value',x=>x[1]]],Object.entries(s.retrieval).map(([k,v])=>[k.replaceAll("_"," "),typeof v==="number"?Number(v.toFixed(3)):v])))+card('Grounding',table([['Metric',x=>x[0]],['Value',x=>x[1]]],Object.entries(s.grounding).map(([k,v])=>[k.replaceAll("_"," "),typeof v==="number"?Number(v.toFixed(3)):v])))+card('Deterministic safety',`<p>Rule accuracy: ${text(s.clinical_rules.accuracy)} | Injection rejection: ${text(s.prompt_injection)} | Safety checks: ${text(s.safety)}</p>`);}
$('refresh-updates').onclick=()=>updates().catch(e=>message(e.message,true));
$('refresh-status').onclick=()=>status().catch(e=>message(e.message,true));
async function initialize(){const me=await request('/v1/me');roles=me.roles;$('session-state').textContent=me.development?'Development demo | '+roles.join(', '):roles.join(', ');await Promise.all([demos(),patients(),sources()]);const params=new URLSearchParams(window.location.search);if(params.has('smart')){const result=await request(`/v1/smart/${encodeURIComponent(params.get('smart'))}/import`,{method:'POST'});history.replaceState({},'', '/');selectedPatient=result.patient_id;$('context-source').textContent=result.context_source;$('context-source').classList.remove('hidden');await patients();}}
$('connect').onclick=()=>{accessToken=$('session-token').value;$('session-token').value='';initialize().catch(e=>message(e.message,true));};
initialize().catch(e=>message(`Connect an authorized session: ${e.message}`,true));
