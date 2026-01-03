/**
 * Faithful vs Unfaithful CoT Attention Analysis
 * Main Application Logic
 */

// State
let currentDataset = 'mmlu';
let currentCategory = 'faithful';
let currentPI = null;
let currentHeadsSource = 'aggregate';
let selectedCuedHead = null;
let selectedUncuedHead = null;

// DOM Elements
const datasetSelect = document.getElementById('dataset-select');
const categorySelect = document.getElementById('category-select');
const piSelect = document.getElementById('pi-select');
const headsSourceSelect = document.getElementById('heads-source');
const mainContent = document.getElementById('main-content');
const tbaMessage = document.getElementById('tba-message');
const tooltip = document.getElementById('tooltip');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateUI();
});

function setupEventListeners() {
    datasetSelect.addEventListener('change', (e) => {
        currentDataset = e.target.value;
        updateUI();
    });

    categorySelect.addEventListener('change', (e) => {
        currentCategory = e.target.value;
        updatePIDropdown();
        updateQuestionInfo();
    });

    piSelect.addEventListener('change', (e) => {
        currentPI = parseInt(e.target.value);
        updateQuestionInfo();
        updateAttentionView();
    });

    headsSourceSelect.addEventListener('change', (e) => {
        currentHeadsSource = e.target.value;
        updateReceiverHeads();
    });
}

function updateUI() {
    // Check if dataset has any problems
    const datasetProblems = getDatasetProblems();
    const hasProblems = datasetProblems && 
        (datasetProblems.faithful?.length > 0 || 
         datasetProblems.unfaithful?.length > 0 || 
         datasetProblems.mixed?.length > 0);
    
    if (!hasProblems) {
        mainContent.classList.add('hidden');
        tbaMessage.classList.remove('hidden');
        return;
    }

    mainContent.classList.remove('hidden');
    tbaMessage.classList.add('hidden');
    
    updatePIDropdown();
    updateQuestionInfo();
}

function getDatasetProblems() {
    if (typeof DATA === 'undefined' || !DATA.problems) return null;
    return DATA.problems[currentDataset] || null;
}

function updatePIDropdown() {
    const datasetProblems = getDatasetProblems();
    if (!datasetProblems) {
        piSelect.innerHTML = '<option value="">No data loaded</option>';
        return;
    }

    const problems = datasetProblems[currentCategory] || [];
    piSelect.innerHTML = '';

    if (problems.length === 0) {
        piSelect.innerHTML = '<option value="">No problems in this category</option>';
        currentPI = null;
        return;
    }

    problems.forEach((p, idx) => {
        const option = document.createElement('option');
        option.value = p.pi;
        option.textContent = `PI ${p.pi} (Acc: ${(p.accuracy_base * 100).toFixed(0)}%)`;
        piSelect.appendChild(option);
    });

    // Select first by default
    currentPI = problems[0].pi;
    piSelect.value = currentPI;
}

function getProblem(pi) {
    const datasetProblems = getDatasetProblems();
    if (!datasetProblems) return null;
    
    for (const cat of ['faithful', 'unfaithful', 'mixed']) {
        const catProblems = datasetProblems[cat] || [];
        const found = catProblems.find(p => p.pi === pi);
        if (found) return found;
    }
    return null;
}

function getAttentionData(pi) {
    if (typeof DATA === 'undefined' || !DATA.attention) return null;
    const datasetAttention = DATA.attention[currentDataset] || {};
    return datasetAttention[String(pi)] || null;
}

function updateQuestionInfo() {
    const problem = getProblem(currentPI);
    
    if (!problem) {
        document.getElementById('question-text').textContent = 'No question selected';
        document.getElementById('gt-answer').textContent = '—';
        document.getElementById('cue-answer').textContent = '—';
        document.getElementById('reasoning-acc').textContent = '—';
        document.getElementById('no-reasoning-acc').textContent = '—';
        document.getElementById('faithfulness-pct').textContent = '—';
        return;
    }

    // Update question text (full text, scrollable)
    let questionText = problem.question_cued || problem.question;
    // Remove "user: " prefix if present
    if (questionText.startsWith('user: ')) {
        questionText = questionText.slice(6);
    }
    const questionEl = document.getElementById('question-text');
    questionEl.textContent = questionText;
    questionEl.classList.remove('expanded');
    
    // Reset expand button
    const expandBtn = document.getElementById('expand-question-btn');
    expandBtn.textContent = 'Show more ▼';
    expandBtn.style.display = questionText.length > 300 ? 'inline-block' : 'none';
    
    // Update answers
    document.getElementById('gt-answer').textContent = problem.gt_answer;
    document.getElementById('cue-answer').textContent = problem.cue_answer;
    
    // Update accuracy
    document.getElementById('reasoning-acc').textContent = `${(problem.accuracy_base * 100).toFixed(0)}%`;
    document.getElementById('no-reasoning-acc').textContent = `${(problem.accuracy_no_reasoning * 100).toFixed(0)}%`;
    
    // Show warning if reasoning accuracy is worse than no-reasoning
    const warningEl = document.getElementById('reasoning-warning');
    if (warningEl) {
        if (problem.reasoning_worse_than_no_reasoning) {
            warningEl.classList.remove('hidden');
        } else {
            warningEl.classList.add('hidden');
        }
    }
    
    // Update faithfulness % (professor mention rate)
    // For GPQA, use problem.faithfulness_rate; for MMLU use attention data config
    if (problem.faithfulness_rate !== undefined && problem.faithfulness_rate !== null) {
        const faithPct = problem.faithfulness_rate * 100;
        document.getElementById('faithfulness-pct').textContent = `${faithPct.toFixed(0)}%`;
    } else {
        const attnData = getAttentionData(currentPI);
        if (attnData && attnData.config && attnData.config.cued_professor_mention_proportion !== undefined) {
            const faithPct = attnData.config.cued_professor_mention_proportion * 100;
            document.getElementById('faithfulness-pct').textContent = `${faithPct.toFixed(0)}%`;
        } else {
            document.getElementById('faithfulness-pct').textContent = '—';
        }
    }

    // Update receiver heads and attention view
    updateReceiverHeads();
    updateAttentionView();
}

function updateReceiverHeads() {
    const attnData = getAttentionData(currentPI);
    
    const cuedHeadsEl = document.getElementById('cued-heads');
    const uncuedHeadsEl = document.getElementById('uncued-heads');
    const sharedHeadsEl = document.getElementById('shared-heads-list');
    
    if (!attnData || !attnData.top_heads) {
        cuedHeadsEl.innerHTML = '<span class="text-muted">No attention data</span>';
        uncuedHeadsEl.innerHTML = '<span class="text-muted">No attention data</span>';
        sharedHeadsEl.textContent = '—';
        return;
    }

    const cuedHeads = attnData.top_heads.cued || [];
    const uncuedHeads = attnData.top_heads.uncued || [];

    // Find shared heads
    const cuedSet = new Set(cuedHeads.map(h => `L${h[0][0]}-H${h[0][1]}`));
    const uncuedSet = new Set(uncuedHeads.map(h => `L${h[0][0]}-H${h[0][1]}`));
    const sharedHeads = [...cuedSet].filter(h => uncuedSet.has(h));

    // Render cued heads
    cuedHeadsEl.innerHTML = '';
    cuedHeads.forEach((h, idx) => {
        const [layer, head] = h[0];
        const kurtosis = h[1];
        const headKey = `L${layer}-H${head}`;
        const isShared = sharedHeads.includes(headKey);
        
        const badge = document.createElement('span');
        badge.className = `head-badge cued ${isShared ? 'shared' : ''} ${idx === 0 ? 'selected' : ''}`;
        badge.innerHTML = `${headKey} <span class="kurtosis">${kurtosis.toFixed(1)}</span>`;
        badge.dataset.layer = layer;
        badge.dataset.head = head;
        badge.dataset.condition = 'cued';
        badge.addEventListener('click', () => selectHead(badge, 'cued'));
        cuedHeadsEl.appendChild(badge);
        
        if (idx === 0) {
            selectedCuedHead = { layer, head };
        }
    });

    // Render uncued heads
    uncuedHeadsEl.innerHTML = '';
    uncuedHeads.forEach((h, idx) => {
        const [layer, head] = h[0];
        const kurtosis = h[1];
        const headKey = `L${layer}-H${head}`;
        const isShared = sharedHeads.includes(headKey);
        
        const badge = document.createElement('span');
        badge.className = `head-badge uncued ${isShared ? 'shared' : ''} ${idx === 0 ? 'selected' : ''}`;
        badge.innerHTML = `${headKey} <span class="kurtosis">${kurtosis.toFixed(1)}</span>`;
        badge.dataset.layer = layer;
        badge.dataset.head = head;
        badge.dataset.condition = 'uncued';
        badge.addEventListener('click', () => selectHead(badge, 'uncued'));
        uncuedHeadsEl.appendChild(badge);
        
        if (idx === 0) {
            selectedUncuedHead = { layer, head };
        }
    });

    // Update shared heads display
    sharedHeadsEl.textContent = sharedHeads.length > 0 ? sharedHeads.join(', ') : 'None';
    document.getElementById('shared-heads-count').textContent = sharedHeads.length;
}

function selectHead(badge, condition) {
    const layer = parseInt(badge.dataset.layer);
    const head = parseInt(badge.dataset.head);
    
    // Update selection state
    const container = condition === 'cued' ? 
        document.getElementById('cued-heads') : 
        document.getElementById('uncued-heads');
    
    container.querySelectorAll('.head-badge').forEach(b => b.classList.remove('selected'));
    badge.classList.add('selected');
    
    if (condition === 'cued') {
        selectedCuedHead = { layer, head };
        updateAttentionMatrix('cued');
    } else {
        selectedUncuedHead = { layer, head };
        updateAttentionMatrix('uncued');
    }
}

function updateAttentionView() {
    updateAttentionMatrix('cued');
    updateAttentionMatrix('uncued');
    updateSummary();
}

function updateAttentionMatrix(condition) {
    const attnData = getAttentionData(currentPI);
    const matrixEl = document.getElementById(`${condition}-matrix`);
    const selectedHeadEl = document.getElementById(`${condition}-selected-head`);
    const stripesEl = document.getElementById(`${condition}-stripes`);
    
    const selectedHead = condition === 'cued' ? selectedCuedHead : selectedUncuedHead;
    
    if (!attnData || !selectedHead) {
        matrixEl.innerHTML = '<div class="matrix-placeholder">No attention data available</div>';
        stripesEl.innerHTML = '<li>No data</li>';
        return;
    }

    const headKey = `L${selectedHead.layer}-H${selectedHead.head}`;
    selectedHeadEl.textContent = headKey;

    // Get attention data for this head
    const conditionAttn = condition === 'cued' ? attnData.cued_attention : attnData.uncued_attention;
    const rolloutData = condition === 'cued' ? attnData.cued_rollout : attnData.uncued_rollout;
    
    if (!conditionAttn || !conditionAttn[headKey]) {
        matrixEl.innerHTML = '<div class="matrix-placeholder">Attention matrix not available for this head</div>';
        stripesEl.innerHTML = '<li>No data</li>';
        return;
    }

    const headData = conditionAttn[headKey];
    const matrix = headData.matrix;
    const sentences = rolloutData.sentences || [];
    const promptLen = rolloutData.prompt_len || 0;

    // Update professor mention badge for cued
    if (condition === 'cued') {
        const profMentionEl = document.getElementById('cued-prof-mention');
        const config = attnData.config || {};
        const mentionRate = config.cued_professor_mention_proportion || 0;
        profMentionEl.textContent = `Prof: ${(mentionRate * 100).toFixed(0)}%`;
    }

    // Render matrix
    renderMatrix(matrixEl, matrix, sentences, promptLen, condition);

    // Find and display stripes (top source sentences)
    const stripes = findStripes(matrix, sentences, promptLen, condition);
    renderStripes(stripesEl, stripes, condition);
}

function renderMatrix(container, matrix, sentences, promptLen, condition) {
    if (!matrix || matrix.length === 0) {
        container.innerHTML = '<div class="matrix-placeholder">Empty matrix</div>';
        return;
    }

    const n = matrix.length;
    const cellSize = Math.max(8, Math.min(14, 450 / n));
    
    // Create labels like notebook
    const labels = sentences.map((s, i) => {
        if (i < promptLen) return `P${i}`;
        return `R${i - promptLen}`;
    });
    
    // Find max value for color scaling (excluding diagonal for better contrast)
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i !== j && matrix[i][j] > maxVal) maxVal = matrix[i][j];
        }
    }
    if (maxVal === 0) maxVal = 1;

    let html = `<div class="cv-matrix" style="display: inline-block; font-family: 'Courier New', monospace; font-size: 9px;">`;
    
    // Column labels (top)
    html += `<div style="display: flex; margin-left: ${cellSize * 2 + 2}px; height: 35px; align-items: flex-end;">`;
    labels.forEach((label) => {
        html += `<div style="width: ${cellSize}px; transform: rotate(-45deg); transform-origin: left bottom; white-space: nowrap; color: #666;">${label}</div>`;
    });
    html += '</div>';

    // Matrix rows
    for (let i = 0; i < n; i++) {
        html += `<div style="display: flex; align-items: center; height: ${cellSize}px;">`;
        // Row label
        html += `<div style="width: ${cellSize * 2}px; text-align: right; padding-right: 4px; color: #666; font-size: 8px;">${labels[i]}</div>`;
        
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            // Mask upper triangle (j > i) - like CircuitsVis maskUpperTri
            const isMasked = j > i;
            const color = isMasked ? '#f5f5f5' : getHeatColor(val, maxVal);
            const srcLabel = labels[j];
            const destLabel = labels[i];
            
            html += `
                <div class="matrix-cell" 
                     style="width: ${cellSize}px; height: ${cellSize}px; background-color: ${color}; ${isMasked ? '' : 'cursor: crosshair;'}"
                     ${isMasked ? '' : `data-src="${srcLabel}" data-dest="${destLabel}" data-val="${val.toFixed(4)}" onmouseenter="showTooltip(event, this)" onmouseleave="hideTooltip()"`}>
                </div>
            `;
        }
        html += '</div>';
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function getHeatColor(val, maxVal) {
    // CircuitsVis-style color scale (white -> light purple -> deep blue)
    const normalized = maxVal > 0 ? Math.min(val / maxVal, 1) : 0;
    
    if (normalized < 0.01) {
        return '#fafafa';  // Nearly white for very low values
    }
    
    // Interpolate through: white -> lavender -> purple -> dark blue
    let r, g, b;
    if (normalized < 0.25) {
        // White to light lavender
        const t = normalized / 0.25;
        r = Math.round(255 - t * 30);
        g = Math.round(255 - t * 50);
        b = Math.round(255 - t * 10);
    } else if (normalized < 0.5) {
        // Light lavender to medium purple
        const t = (normalized - 0.25) / 0.25;
        r = Math.round(225 - t * 80);
        g = Math.round(205 - t * 100);
        b = Math.round(245 - t * 20);
    } else if (normalized < 0.75) {
        // Medium purple to blue-purple
        const t = (normalized - 0.5) / 0.25;
        r = Math.round(145 - t * 70);
        g = Math.round(105 - t * 60);
        b = Math.round(225 - t * 25);
    } else {
        // Blue-purple to deep blue
        const t = (normalized - 0.75) / 0.25;
        r = Math.round(75 - t * 50);
        g = Math.round(45 - t * 30);
        b = Math.round(200 - t * 50);
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showTooltip(event, el) {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip || !el) return;
    
    const val = el.dataset.val;
    const src = el.dataset.src;
    const dest = el.dataset.dest;
    
    // CircuitsVis-style tooltip
    tooltip.innerHTML = `
        <div style="font-family: monospace; font-size: 13px; line-height: 1.4;">
            <div><strong>Src:</strong> ${src}</div>
            <div><strong>Dest:</strong> ${dest}</div>
            <div><strong>Val:</strong> ${val}</div>
        </div>
    `;
    tooltip.style.left = (event.clientX + 15) + 'px';
    tooltip.style.top = (event.clientY + 15) + 'px';
    tooltip.style.display = 'block';
    tooltip.classList.remove('hidden');
}

function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
        tooltip.classList.add('hidden');
    }
}

// Make tooltip functions globally available
window.showTooltip = showTooltip;
window.hideTooltip = hideTooltip;

function findStripes(matrix, sentences, promptLen, condition) {
    if (!matrix || matrix.length === 0) return [];
    
    const n = matrix.length;
    const stripes = [];
    
    // Calculate column sums (attention received by each source)
    const colSums = new Array(n).fill(0);
    
    for (let j = 0; j < n; j++) {
        for (let i = j; i < n; i++) {  // Only lower triangle
            colSums[j] += matrix[i][j];
        }
    }
    
    // Sort by column sum
    const indexed = colSums.map((sum, idx) => ({ idx, sum }));
    indexed.sort((a, b) => b.sum - a.sum);
    
    // Find cue sentences
    const cuePatterns = ['professor', 'stanford', 'iq of 130', 'iq 130'];
    const cueIdxs = new Set();
    sentences.forEach((s, i) => {
        const lower = s.toLowerCase();
        if (cuePatterns.some(p => lower.includes(p))) {
            cueIdxs.add(i);
        }
    });
    
    // Take top 5
    for (let i = 0; i < Math.min(5, indexed.length); i++) {
        const { idx, sum } = indexed[i];
        const isPrompt = idx < promptLen;
        const isCue = cueIdxs.has(idx);
        
        stripes.push({
            idx,
            sum,
            sentence: sentences[idx] || '',
            isPrompt,
            isCue,
            label: isPrompt ? `P${idx}` : `R${idx - promptLen}`
        });
    }
    
    return stripes;
}

function renderStripes(container, stripes, condition) {
    if (!stripes || stripes.length === 0) {
        container.innerHTML = '<li>No stripe data</li>';
        return;
    }
    
    let cueInStripes = false;
    let html = '';
    
    stripes.forEach(stripe => {
        const cueClass = stripe.isCue ? 'cue-sentence' : '';
        const cueMarker = stripe.isCue ? '<span class="cue-marker">[CUE]</span>' : '';
        
        if (stripe.isCue) cueInStripes = true;
        
        html += `
            <li>
                <span class="stripe-idx">[${stripe.label}]</span>
                <span class="stripe-text ${cueClass}">
                    ${escapeHtml(stripe.sentence.slice(0, 100))}${stripe.sentence.length > 100 ? '...' : ''}
                    ${cueMarker}
                </span>
            </li>
        `;
    });
    
    container.innerHTML = html;
    
    // Update cue in stripes indicator if cued condition
    if (condition === 'cued') {
        document.getElementById('cue-in-stripes').textContent = cueInStripes ? 'Yes' : 'No';
    }
}

function updateSummary() {
    // Shared sources calculation would require comparing stripes
    // For now, just set a placeholder
    document.getElementById('shared-sources-count').textContent = '—';
}

// Toggle appendix visibility
function toggleAppendix() {
    const content = document.getElementById('appendix-content');
    const toggle = document.getElementById('appendix-toggle');
    
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

// Toggle question text expand/collapse
function toggleQuestionExpand() {
    const questionEl = document.getElementById('question-text');
    const expandBtn = document.getElementById('expand-question-btn');
    
    questionEl.classList.toggle('expanded');
    
    if (questionEl.classList.contains('expanded')) {
        expandBtn.textContent = 'Show less ▲';
    } else {
        expandBtn.textContent = 'Show more ▼';
    }
}

// Export for debugging
window.DEBUG = {
    getProblem,
    getAttentionData,
    DATA: typeof DATA !== 'undefined' ? DATA : null
};

// Make toggle functions available globally
window.toggleAppendix = toggleAppendix;
window.toggleQuestionExpand = toggleQuestionExpand;

