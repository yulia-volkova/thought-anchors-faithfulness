/**
 * Faithful vs Unfaithful CoT Attention Analysis
 * Main Application Logic
 */

// State
let currentDataset = 'mmlu';
let selectedCategories = ['faithful', 'unfaithful'];  // Multiselect
let currentPI = null;
let currentHeadsSource = 'aggregate';
let currentAttentionMode = 'full';  // 'full' = with prompt, 'reasoning' = reasoning only
let selectedCuedHead = null;
let selectedUncuedHead = null;
let selectedFvuHead = null;  // For faithful vs unfaithful section

// Allowed PIs for MMLU (only show these)
const ALLOWED_MMLU_PIS = [91, 152, 188, 19, 151, 182, 191];

// DOM Elements
const datasetSelect = document.getElementById('dataset-select');
const categoryCheckboxes = document.querySelectorAll('#category-checkboxes input[type="checkbox"]');
const piSelect = document.getElementById('pi-select');
const headsSourceSelect = document.getElementById('heads-source');
const attentionModeSelect = document.getElementById('attention-mode');
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

    // Category checkboxes (multiselect)
    categoryCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            selectedCategories = Array.from(categoryCheckboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);
            
            // Ensure at least one category is selected
            if (selectedCategories.length === 0) {
                checkbox.checked = true;
                selectedCategories = [checkbox.value];
            }
            
            updatePIDropdown();
            updateQuestionInfo();
        });
    });

    piSelect.addEventListener('change', (e) => {
        currentPI = parseInt(e.target.value);
        // Reset head selections when PI changes
        selectedCuedHead = null;
        selectedUncuedHead = null;
        selectedFvuHead = null;
        updateQuestionInfo();
        updateReceiverHeads();  // This will set new default heads
        updateAttentionView();
    });

    headsSourceSelect.addEventListener('change', (e) => {
        currentHeadsSource = e.target.value;
        updateReceiverHeads();
    });

    attentionModeSelect.addEventListener('change', (e) => {
        currentAttentionMode = e.target.value;
        // Reset head selections when mode changes (different top heads)
        selectedCuedHead = null;
        selectedUncuedHead = null;
        selectedFvuHead = null;
        updateReceiverHeads();
        updateAttentionView();
    });

    // Setup tooltip for receiver heads source info
    const headsSourceInfo = document.getElementById('heads-source-info');
    const headsSourceTooltip = document.getElementById('heads-source-tooltip');
    if (headsSourceInfo && headsSourceTooltip) {
        headsSourceInfo.addEventListener('mouseenter', () => {
            headsSourceTooltip.classList.remove('hidden');
        });
        headsSourceInfo.addEventListener('mouseleave', () => {
            headsSourceTooltip.classList.add('hidden');
        });
        // Also show/hide when hovering over the tooltip itself
        headsSourceTooltip.addEventListener('mouseenter', () => {
            headsSourceTooltip.classList.remove('hidden');
        });
        headsSourceTooltip.addEventListener('mouseleave', () => {
            headsSourceTooltip.classList.add('hidden');
        });
    }
}

function updateUI() {
    // Check if dataset has any problems
    const datasetProblems = getDatasetProblems();
    const hasProblems = datasetProblems && 
        (datasetProblems.faithful?.length > 0 || 
         datasetProblems.unfaithful?.length > 0);
    
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

    // Combine problems from all selected categories
    let problems = [];
    selectedCategories.forEach(cat => {
        const catProblems = datasetProblems[cat] || [];
        problems = problems.concat(catProblems.map(p => ({...p, category: cat})));
    });
    
    // Filter MMLU to only allowed PIs
    if (currentDataset === 'mmlu') {
        problems = problems.filter(p => ALLOWED_MMLU_PIS.includes(p.pi));
    }
    
    // Sort by PI
    problems.sort((a, b) => a.pi - b.pi);
    
    piSelect.innerHTML = '';

    if (problems.length === 0) {
        piSelect.innerHTML = '<option value="">No problems in selected categories</option>';
        currentPI = null;
        return;
    }

    problems.forEach((p, idx) => {
        const option = document.createElement('option');
        option.value = p.pi;
        option.textContent = `PI ${p.pi} (Acc: ${(p.accuracy_base * 100).toFixed(0)}%)`;
        // Color-code: green for faithful, red-orange for unfaithful
        if (p.category === 'faithful') {
            option.style.color = '#166534';  // Green
        } else {
            option.style.color = '#991b1b';  // Red
        }
        piSelect.appendChild(option);
    });

    // Select first by default
    currentPI = problems[0].pi;
    piSelect.value = currentPI;
}

function getProblem(pi) {
    const datasetProblems = getDatasetProblems();
    if (!datasetProblems) return null;
    
    for (const cat of ['faithful', 'unfaithful']) {
        const catProblems = datasetProblems[cat] || [];
        const found = catProblems.find(p => p.pi === pi);
        if (found) return {...found, category: cat};
    }
    return null;
}

function updateCategoryBadges(category) {
    const badges = [
        document.getElementById('pi-category-badge'),
        document.getElementById('heads-category-badge'),
        document.getElementById('attention-category-badge')
    ];
    
    badges.forEach(badge => {
        if (!badge) return;
        badge.className = 'category-badge';
        if (category === 'faithful') {
            badge.textContent = 'Faithful';
            badge.classList.add('faithful');
        } else if (category === 'unfaithful') {
            badge.textContent = 'Unfaithful';
            badge.classList.add('unfaithful');
        } else {
            badge.textContent = '';
        }
    });
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
        updateCategoryBadges(null);
        return;
    }
    
    // Update category badges
    updateCategoryBadges(problem.category);

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

// Helper function to check if a head has cue mentions in its top source sentences
function headHasCueInTopSources(attnData, layer, head, condition) {
    if (!attnData) return false;
    
    const headKey = `L${layer}-H${head}`;
    
    // Get attention data for this head (use reasoning-only version if in reasoning mode)
    let conditionAttn;
    if (currentAttentionMode === 'reasoning') {
        conditionAttn = condition === 'cued' ? attnData.cued_attention_reasoning : attnData.uncued_attention_reasoning;
    } else {
        conditionAttn = condition === 'cued' ? attnData.cued_attention : attnData.uncued_attention;
    }
    // Fall back to full attention if reasoning version not available
    if (!conditionAttn) {
        conditionAttn = condition === 'cued' ? attnData.cued_attention : attnData.uncued_attention;
    }
    
    if (!conditionAttn || !conditionAttn[headKey]) return false;
    
    const rolloutData = condition === 'cued' ? attnData.cued_rollout : attnData.uncued_rollout;
    if (!rolloutData) return false;
    
    const headData = conditionAttn[headKey];
    const matrix = headData.matrix;
    const allSentences = rolloutData.sentences || [];
    const fullPromptLen = Math.min(rolloutData.prompt_len || 0, allSentences.length);
    
    // In reasoning mode, only show reasoning sentences (no prompt)
    let sentences, promptLen;
    if (currentAttentionMode === 'reasoning') {
        sentences = allSentences.slice(fullPromptLen);
        promptLen = 0;
    } else {
        sentences = allSentences;
        promptLen = fullPromptLen;
    }
    
    // Find top source sentences (stripes) and check if any contain cues
    const stripes = findStripes(matrix, sentences, promptLen, condition);
    return stripes.some(stripe => stripe.isCue);
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

    // Get heads based on attention mode (full vs reasoning only)
    const cuedKey = currentAttentionMode === 'reasoning' ? 'cued_reasoning' : 'cued';
    const uncuedKey = currentAttentionMode === 'reasoning' ? 'uncued_reasoning' : 'uncued';
    const cuedHeads = attnData.top_heads[cuedKey] || attnData.top_heads.cued || [];
    const uncuedHeads = attnData.top_heads[uncuedKey] || attnData.top_heads.uncued || [];

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
        const hasCueInTopSources = headHasCueInTopSources(attnData, layer, head, 'cued');
        
        const badge = document.createElement('span');
        badge.className = `head-badge cued ${isShared ? 'shared' : ''} ${idx === 0 ? 'selected' : ''} ${hasCueInTopSources ? 'has-cue' : ''}`;
        badge.innerHTML = `${headKey} <span class="kurtosis">${kurtosis.toFixed(1)}</span>`;
        badge.dataset.layer = layer;
        badge.dataset.head = head;
        badge.dataset.condition = 'cued';
        badge.addEventListener('click', () => selectHead(badge, 'cued'));
        cuedHeadsEl.appendChild(badge);
        
        if (idx === 0) {
            selectedCuedHead = { layer, head };
            selectedFvuHead = { layer, head };  // Also set FvU head to match cued
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
    const headKey = `L${layer}-H${head}`;
    
    const cuedContainer = document.getElementById('cued-heads');
    const uncuedContainer = document.getElementById('uncued-heads');
    
    // Check if this head exists in the other column (is shared)
    const otherContainer = condition === 'cued' ? uncuedContainer : cuedContainer;
    const matchingBadgeInOther = Array.from(otherContainer.querySelectorAll('.head-badge')).find(b => 
        parseInt(b.dataset.layer) === layer && parseInt(b.dataset.head) === head
    );
    
    // Clear selections in current container
    const container = condition === 'cued' ? cuedContainer : uncuedContainer;
    container.querySelectorAll('.head-badge').forEach(b => b.classList.remove('selected'));
    badge.classList.add('selected');
    
    // If shared, also select in the other container
    if (matchingBadgeInOther) {
        otherContainer.querySelectorAll('.head-badge').forEach(b => b.classList.remove('selected'));
        matchingBadgeInOther.classList.add('selected');
        
        // Update both heads
        selectedCuedHead = { layer, head };
        selectedUncuedHead = { layer, head };
        selectedFvuHead = { layer, head };
        
        // Update both matrices
        updateAttentionMatrix('cued');
        updateAttentionMatrix('uncued');
        updateFaithfulVsUnfaithful();
    } else {
        // Not shared - only update the clicked column
        if (condition === 'cued') {
            selectedCuedHead = { layer, head };
            selectedFvuHead = { layer, head };
            updateAttentionMatrix('cued');
            updateFaithfulVsUnfaithful();
        } else {
            selectedUncuedHead = { layer, head };
            updateAttentionMatrix('uncued');
        }
    }
}

function updateAttentionView() {
    updateAttentionMatrix('cued');
    updateAttentionMatrix('uncued');
    updateFaithfulVsUnfaithful();
    updateSummary();
}

// ================================
// Faithful vs Unfaithful Section
// ================================
function updateFaithfulVsUnfaithful() {
    const fvuSection = document.getElementById('fvu-section');
    const fvuNotAvailable = document.getElementById('fvu-not-available');
    const fvuComparison = document.getElementById('fvu-comparison');
    
    const attnData = getAttentionData(currentPI);
    
    // Check if FvU data is available
    if (!attnData || !attnData.has_faithful_vs_unfaithful) {
        fvuSection.classList.remove('hidden');
        fvuNotAvailable.classList.remove('hidden');
        fvuNotAvailable.innerHTML = `<p>⚠️ Faithful vs Unfaithful comparison not available for this problem. 
            This requires both a cued rollout that mentions the professor AND one that doesn't.</p>`;
        fvuComparison.classList.add('hidden');
        return;
    }
    
    // Check if consistently faithful (no unfaithful rollout could be generated)
    if (attnData.consistently_faithful) {
        const rate = attnData.generation_faithful_rate || 1.0;
        fvuSection.classList.remove('hidden');
        fvuNotAvailable.classList.remove('hidden');
        fvuNotAvailable.innerHTML = `
            <p>🎯 <strong>Consistently Faithful:</strong> This problem produces faithful CoT rollouts 
            <strong>${(rate * 100).toFixed(0)}%</strong> of the time when cued. 
            No unfaithful rollout could be generated after 5 attempts.</p>
            <p style="margin-top: 0.5rem; font-size: 0.9em; opacity: 0.8;">
                The faithful attention pattern is shown below (no unfaithful comparison available).
            </p>`;
        fvuComparison.classList.remove('hidden');
        
        // Hide the unfaithful panel, show faithful
        const unfaithfulPanel = document.querySelector('.unfaithful-panel');
        const faithfulPanel = document.querySelector('.faithful-panel');
        if (unfaithfulPanel) unfaithfulPanel.classList.add('hidden');
        if (faithfulPanel) faithfulPanel.classList.remove('hidden');
        
        // Show only faithful
        if (!selectedFvuHead && selectedCuedHead) {
            selectedFvuHead = selectedCuedHead;
        }
        if (!selectedFvuHead) {
            const cuedKey = currentAttentionMode === 'reasoning' ? 'cued_reasoning' : 'cued';
            const cuedHeads = attnData.top_heads[cuedKey] || attnData.top_heads.cued || [];
            if (cuedHeads.length > 0) {
                selectedFvuHead = { layer: cuedHeads[0][0][0], head: cuedHeads[0][0][1] };
            }
        }
        updateFvuMatrix('faithful', attnData, selectedFvuHead);
        return;
    }
    
    // Check if consistently unfaithful (no faithful rollout could be generated)
    if (attnData.consistently_unfaithful) {
        const rate = attnData.generation_faithful_rate || 0.0;
        fvuSection.classList.remove('hidden');
        fvuNotAvailable.classList.remove('hidden');
        fvuNotAvailable.innerHTML = `
            <p>🔇 <strong>Consistently Unfaithful:</strong> This problem produces unfaithful CoT rollouts 
            <strong>${((1 - rate) * 100).toFixed(0)}%</strong> of the time when cued (never mentions the cue). 
            No faithful rollout could be generated after 5 attempts.</p>
            <p style="margin-top: 0.5rem; font-size: 0.9em; opacity: 0.8;">
                The unfaithful attention pattern is shown below (no faithful comparison available).
            </p>`;
        fvuComparison.classList.remove('hidden');
        
        // Hide the faithful panel, show unfaithful
        const faithfulPanel = document.querySelector('.faithful-panel');
        const unfaithfulPanel = document.querySelector('.unfaithful-panel');
        if (faithfulPanel) faithfulPanel.classList.add('hidden');
        if (unfaithfulPanel) unfaithfulPanel.classList.remove('hidden');
        
        // Show only unfaithful
        if (!selectedFvuHead && selectedCuedHead) {
            selectedFvuHead = selectedCuedHead;
        }
        if (!selectedFvuHead) {
            const cuedKey = currentAttentionMode === 'reasoning' ? 'cued_reasoning' : 'cued';
            const cuedHeads = attnData.top_heads[cuedKey] || attnData.top_heads.cued || [];
            if (cuedHeads.length > 0) {
                selectedFvuHead = { layer: cuedHeads[0][0][0], head: cuedHeads[0][0][1] };
            }
        }
        updateFvuMatrix('unfaithful', attnData, selectedFvuHead);
        return;
    }
    
    // Show the full comparison (both faithful and unfaithful)
    fvuSection.classList.remove('hidden');
    fvuNotAvailable.classList.add('hidden');
    fvuComparison.classList.remove('hidden');
    
    // Show the unfaithful panel
    const unfaithfulPanel = document.querySelector('.unfaithful-panel');
    if (unfaithfulPanel) unfaithfulPanel.classList.remove('hidden');
    
    // Use the same head as the cued rollout for consistency
    if (!selectedFvuHead && selectedCuedHead) {
        selectedFvuHead = selectedCuedHead;
    }
    
    if (!selectedFvuHead) {
        // Default to first cued head (respecting attention mode)
        const cuedKey = currentAttentionMode === 'reasoning' ? 'cued_reasoning' : 'cued';
        const cuedHeads = attnData.top_heads[cuedKey] || attnData.top_heads.cued || [];
        if (cuedHeads.length > 0) {
            selectedFvuHead = { layer: cuedHeads[0][0][0], head: cuedHeads[0][0][1] };
        }
    }
    
    // Update both faithful and unfaithful matrices
    updateFvuMatrix('faithful', attnData, selectedFvuHead);
    updateFvuMatrix('unfaithful', attnData, selectedFvuHead);
}

function updateFvuMatrix(type, attnData, selectedHead) {
    const matrixEl = document.getElementById(`${type}-matrix`);
    const headBadgeEl = document.getElementById(`fvu-${type}-head`);
    const stripesEl = document.getElementById(`${type}-stripes`);
    
    if (!selectedHead) {
        matrixEl.innerHTML = '<div class="matrix-placeholder">No head selected</div>';
        stripesEl.innerHTML = '<li>No data</li>';
        return;
    }
    
    const headKey = `L${selectedHead.layer}-H${selectedHead.head}`;
    headBadgeEl.textContent = headKey;
    
    // Get the appropriate data (use reasoning-only version if in reasoning mode)
    let attentionData;
    if (currentAttentionMode === 'reasoning') {
        attentionData = type === 'faithful' ? attnData.faithful_attention_reasoning : attnData.unfaithful_attention_reasoning;
    } else {
        attentionData = type === 'faithful' ? attnData.faithful_attention : attnData.unfaithful_attention;
    }
    // Fall back to full attention if reasoning version not available
    if (!attentionData) {
        attentionData = type === 'faithful' ? attnData.faithful_attention : attnData.unfaithful_attention;
    }
    const rolloutData = type === 'faithful' ? attnData.faithful_rollout : attnData.unfaithful_rollout;
    
    if (!attentionData || !attentionData[headKey]) {
        matrixEl.innerHTML = '<div class="matrix-placeholder">Attention matrix not available for this head</div>';
        stripesEl.innerHTML = '<li>No data</li>';
        return;
    }
    
    const headData = attentionData[headKey];
    const matrix = headData.matrix;
    const allSentences = rolloutData.sentences || [];
    // Cap prompt_len to number of sentences (fix for bad data)
    const fullPromptLen = Math.min(rolloutData.prompt_len || 0, allSentences.length);

    // In reasoning mode, only show reasoning sentences (no prompt)
    let sentences, promptLen;
    if (currentAttentionMode === 'reasoning') {
        sentences = allSentences.slice(fullPromptLen);  // Only reasoning sentences
        promptLen = 0;  // No prompt in reasoning-only view
    } else {
        sentences = allSentences;
        promptLen = fullPromptLen;
    }

    // Render matrix
    renderMatrix(matrixEl, matrix, sentences, promptLen, type);

    // Find and display stripes
    const stripes = findStripes(matrix, sentences, promptLen, type);
    renderStripes(stripesEl, stripes, type);
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

    // Get attention data for this head (use reasoning-only version if in reasoning mode)
    let conditionAttn;
    if (currentAttentionMode === 'reasoning') {
        conditionAttn = condition === 'cued' ? attnData.cued_attention_reasoning : attnData.uncued_attention_reasoning;
    } else {
        conditionAttn = condition === 'cued' ? attnData.cued_attention : attnData.uncued_attention;
    }
    // Fall back to full attention if reasoning version not available
    if (!conditionAttn) {
        conditionAttn = condition === 'cued' ? attnData.cued_attention : attnData.uncued_attention;
    }
    const rolloutData = condition === 'cued' ? attnData.cued_rollout : attnData.uncued_rollout;
    
    if (!conditionAttn || !conditionAttn[headKey]) {
        matrixEl.innerHTML = '<div class="matrix-placeholder">Attention matrix not available for this head</div>';
        stripesEl.innerHTML = '<li>No data</li>';
        return;
    }

    const headData = conditionAttn[headKey];
    const matrix = headData.matrix;
    const allSentences = rolloutData.sentences || [];
    // Cap prompt_len to number of sentences (fix for bad data)
    const fullPromptLen = Math.min(rolloutData.prompt_len || 0, allSentences.length);

    // Update professor mention badge for cued
    if (condition === 'cued') {
        const profMentionEl = document.getElementById('cued-prof-mention');
        const config = attnData.config || {};
        const mentionRate = config.cued_professor_mention_proportion || 0;
        profMentionEl.textContent = `Prof: ${(mentionRate * 100).toFixed(0)}%`;
    }

    // In reasoning mode, only show reasoning sentences (no prompt)
    let sentences, promptLen;
    if (currentAttentionMode === 'reasoning') {
        sentences = allSentences.slice(fullPromptLen);  // Only reasoning sentences
        promptLen = 0;  // No prompt in reasoning-only view
    } else {
        sentences = allSentences;
        promptLen = fullPromptLen;
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
    
    // For large matrices (GPQA), group sentences
    const MAX_DISPLAY = 30;
    const shouldGroup = n > MAX_DISPLAY;
    const groupSize = shouldGroup ? Math.ceil(n / MAX_DISPLAY) : 1;
    const displayN = shouldGroup ? Math.ceil(n / groupSize) : n;
    
    // Group the matrix if needed
    let displayMatrix, displayLabels;
    if (shouldGroup) {
        displayMatrix = [];
        displayLabels = [];
        
        for (let gi = 0; gi < displayN; gi++) {
            const rowStart = gi * groupSize;
            const rowEnd = Math.min((gi + 1) * groupSize, n);
            
            // Create label for this group
            const startLabel = rowStart < promptLen ? `P${rowStart}` : `R${rowStart - promptLen}`;
            const endIdx = rowEnd - 1;
            const endLabel = endIdx < promptLen ? `P${endIdx}` : `R${endIdx - promptLen}`;
            displayLabels.push(rowEnd - rowStart > 1 ? `${startLabel}-${endLabel.replace(/[PR]/, '')}` : startLabel);
            
            const row = [];
            for (let gj = 0; gj < displayN; gj++) {
                const colStart = gj * groupSize;
                const colEnd = Math.min((gj + 1) * groupSize, n);
                
                // Average the values in this block
                let sum = 0, count = 0;
                for (let i = rowStart; i < rowEnd; i++) {
                    for (let j = colStart; j < colEnd; j++) {
                        sum += matrix[i][j];
                        count++;
                    }
                }
                row.push(count > 0 ? sum / count : 0);
            }
            displayMatrix.push(row);
        }
    } else {
        displayMatrix = matrix;
        displayLabels = sentences.map((s, i) => {
            if (i < promptLen) return `P${i}`;
            return `R${i - promptLen}`;
        });
    }
    
    // Calculate cell size to fill container (target ~380px for the matrix)
    const targetSize = 360;
    const cellSize = Math.max(8, Math.floor(targetSize / displayN));
    
    // Find max value for color scaling
    let maxVal = 0;
    for (let i = 0; i < displayN; i++) {
        for (let j = 0; j < displayN; j++) {
            if (i !== j && displayMatrix[i][j] > maxVal) maxVal = displayMatrix[i][j];
        }
    }
    if (maxVal === 0) maxVal = 1;

    const fontSize = displayN > 25 ? 7 : (displayN > 15 ? 8 : 9);
    const labelWidth = displayN > 25 ? cellSize * 3.5 : cellSize * 2.5;
    
    let html = `<div class="cv-matrix" style="display: inline-block; font-family: 'Courier New', monospace; font-size: ${fontSize}px;">`;
    
    // Column labels (top)
    html += `<div style="display: flex; margin-left: ${labelWidth + 2}px; height: 45px; align-items: flex-end;">`;
    displayLabels.forEach((label) => {
        html += `<div style="width: ${cellSize}px; transform: rotate(-45deg); transform-origin: left bottom; white-space: nowrap; color: #666; font-size: ${fontSize}px;">${label}</div>`;
    });
    html += '</div>';

    // Matrix rows
    for (let i = 0; i < displayN; i++) {
        html += `<div style="display: flex; align-items: center; height: ${cellSize}px;">`;
        // Row label
        html += `<div style="width: ${labelWidth}px; text-align: right; padding-right: 4px; color: #666; font-size: ${fontSize}px;">${displayLabels[i]}</div>`;
        
        for (let j = 0; j < displayN; j++) {
            const val = displayMatrix[i][j];
            // Mask upper triangle
            const isMasked = j > i;
            const color = isMasked ? '#f5f5f5' : getHeatColor(val, maxVal);
            
            html += `
                <div class="matrix-cell" 
                     style="width: ${cellSize}px; height: ${cellSize}px; background-color: ${color}; ${isMasked ? '' : 'cursor: crosshair;'}"
                     ${isMasked ? '' : `data-src="${displayLabels[j]}" data-dest="${displayLabels[i]}" data-val="${val.toFixed(4)}" onmouseenter="showTooltip(event, this)" onmouseleave="hideTooltip()"`}>
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
    
    // Take top 5 by attention
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
    
    // Sort by order of appearance (idx)
    stripes.sort((a, b) => a.idx - b.idx);
    
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
        
        // Build class string properly (trim to avoid extra spaces)
        const classStr = cueClass ? `stripe-text ${cueClass}` : 'stripe-text';
        
        html += `
            <li>
                <span class="stripe-idx">[${stripe.label}]</span>
                <span class="${classStr}">
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

// Toggle sidenote expand/collapse
function toggleSidenote(id) {
    const content = document.getElementById(`${id}-content`);
    const toggle = document.getElementById(`${id}-toggle`);

    content.classList.toggle('collapsed');
    toggle.classList.toggle('expanded');
}

// Make toggle functions available globally
window.toggleAppendix = toggleAppendix;
window.toggleQuestionExpand = toggleQuestionExpand;
window.toggleSidenote = toggleSidenote;

