// Exercise the HTML client's Ask Evidence renderer without network or a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'frontend', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, 'client script exists');
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, { value: '', innerHTML: '', textContent: '', onclick: null });
  return elements.get(id);
};
let evidenceResponse = null;
const context = {
  document: { getElementById: get, querySelectorAll: () => [] },
  fetch: async url => ({
    ok: true,
    json: async () => url.endsWith('/v1/evidence/query') ? evidenceResponse : [],
  }),
  console,
};
vm.runInNewContext(script, context, { filename: 'frontend/index.html' });

const evidence = {
  evidence_unit_id: 'unit-1', publisher: 'Publisher', canonical_title: 'Guideline',
  version_id: 'version-1', updated_at: '2026-01-01', jurisdiction: 'UK',
  lifecycle_status: 'current', recommendation_id: '1.4.24', page_start: 2, page_end: 2,
  supporting_excerpt: '<script>untrusted</script>', canonical_source_url: 'https://example.org/source',
};

async function ask(intent, result) {
  evidenceResponse = result;
  get('intent').value = intent;
  get('question').value = 'What does the source say?';
  await get('ask').onclick();
  return get('answer').innerHTML;
}

(async () => {
  let rendered = await ask('clinical_guidance', {
    status: 'grounded', answer_text: 'Annual review.',
    claims: [{ claim_id: 'c1', text: 'Annual review.', evidence_ids: ['unit-1'], support_status: 'supported',
      verification_passages: { 'unit-1': 'Exact checked passage.' } }],
    conflicts: [], evidence: [evidence],
  });
  assert.match(rendered, /Claim · supported/);
  assert.match(rendered, /version-1/);
  assert.match(rendered, /Exact checked passage/);
  assert.doesNotMatch(rendered, /untrusted/);
  evidenceResponse.claims[0].verification_passages['unit-1'] = '<script>untrusted</script>';
  rendered = await get('ask').onclick().then(() => get('answer').innerHTML);
  assert.match(rendered, /&lt;script&gt;untrusted&lt;\/script&gt;/);
  assert.doesNotMatch(rendered, /<script>untrusted<\/script>/);

  rendered = await ask('clinical_guidance', {
    status: 'conflict', answer_text: 'Evidence differs.', claims: [],
    conflicts: [{ conflict_id: 'x', description: 'Material difference.', evidence_ids: ['unit-1'] }],
    evidence: [evidence],
  });
  assert.match(rendered, /Evidence differs/);
  assert.match(rendered, /Material difference/);

  rendered = await ask('clinical_guidance', {
    status: 'abstained', answer_text: '', claims: [], conflicts: [], evidence: [],
  });
  assert.match(rendered, /does not support a reliable answer/);
  assert.doesNotMatch(rendered, /Guideline/);

  rendered = await ask('historical', {
    status: 'abstained', answer_text: '', claims: [], conflicts: [], evidence: [],
  });
  assert.match(rendered, /Historical \/ superseded evidence/);
  console.log('Ask Evidence UI smoke passed: claims, escaping, conflict, abstention, historical warning');
})().catch(error => { console.error(error); process.exitCode = 1; });
