// pricing for claude-sonnet-4 (USD per million tokens) — adjust as needed
const PRICE_INPUT_PER_M  = 1.00;
const PRICE_OUTPUT_PER_M = 5.00;
const CHARS_PER_TOKEN    = 4; // rough estimate

const promptInput = document.getElementById('promptInput');
const runBtn      = document.getElementById('runBtn');
const transcript  = document.getElementById('transcript');
const emptyEl     = document.getElementById('empty');

const elElapsed = document.getElementById('elapsed');
const elChars   = document.getElementById('chars');
const elTokens  = document.getElementById('tokens');
const elCost    = document.getElementById('cost');
const elSteps   = document.getElementById('steps');

let elapsedTimer = null;
let startTime    = 0;
let conversationHistory = []; // for multi-turn

// ─── live char/token preview as user types ───
promptInput.addEventListener('input', () => {
    const c = promptInput.value.length;
    const t = Math.ceil(c / CHARS_PER_TOKEN);
    const cost = (t / 1_000_000) * PRICE_INPUT_PER_M;
    if (!isRunning) {
        elChars.textContent  = c;
        elTokens.textContent = t;
        elCost.textContent   = '$' + cost.toFixed(4);
    }
});

// ─── steps ───
const STEPS = ['parsing', 'sending', 'receiving', 'done'];
function renderSteps(activeIdx = -1) {
    elSteps.innerHTML = '';
    STEPS.forEach((label, i) => {
        const div = document.createElement('div');
        div.className = 'step';
        if (i < activeIdx) div.classList.add('done');
        if (i === activeIdx) div.classList.add('active');
        div.innerHTML = `<span class="dot"></span><span>${i+1}. ${label}</span>`;
        elSteps.appendChild(div);
    });
}

function resetSteps() {
    elSteps.innerHTML = '<div class="step"><span class="dot"></span><span>idle</span></div>';
}

// ─── transcript ───
function addMessage(role, text) {
    if (emptyEl && emptyEl.parentNode) emptyEl.remove();
    const msg = document.createElement('div');
    msg.className = 'msg ' + role;
    msg.innerHTML = `<span class="role">${role === 'user' ? '› you' : '› nanocontrol'}</span><div class="body"></div>`;
    msg.querySelector('.body').textContent = text;
    transcript.appendChild(msg);
    transcript.scrollTop = transcript.scrollHeight;
    return msg.querySelector('.body');
}

// ─── execution ───
let isRunning = false;

async function execute() {
    const prompt = promptInput.value.trim();
    if (!prompt || isRunning) return;

    isRunning = true;
    runBtn.disabled = true;
    promptInput.disabled = true;

    addMessage('user', prompt);
    promptInput.value = '';
    conversationHistory.push({ role: 'user', content: prompt });

    // start timer
    startTime = performance.now();
    elapsedTimer = setInterval(() => {
      const s = (performance.now() - startTime) / 1000;
      elElapsed.textContent = s.toFixed(2) + 's';
    }, 50);

    // step 1: parsing
    renderSteps(0);
    await sleep(150);

    // input metrics
    const inputChars  = prompt.length;
    const inputTokens = Math.ceil(inputChars / CHARS_PER_TOKEN);
    elChars.textContent = inputChars;
    elTokens.textContent = inputTokens;

    // step 2: sending
    renderSteps(1);

    const responseBody = addMessage('assistant', '');
    responseBody.innerHTML = '<span class="cursor"></span>';

    let outputText = '';
    let outputTokens = 0;
    let inTok  = inputTokens;
    let outTok = 0;
    let tools  = 0;

    try {
        const response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: conversationHistory
          })
        });

        // step 3: receiving
        renderSteps(2);

        const data = await response.json();
        outputText = (data.content || [])
          .map(b => b.type === 'text' ? b.text : '')
          .filter(Boolean)
          .join('\n');
              
        const usage = data.usage || {};
        inTok  = usage.input_tokens  || inputTokens;
        outTok = usage.output_tokens || 0;
        tools  = usage.tool_calls    || 0;
        outputTokens = outTok;
              
        console.log(`[cost breakdown]
          input:  ${inTok} tokens  → $${((inTok / 1_000_000) * PRICE_INPUT_PER_M).toFixed(6)}
          output: ${outTok} tokens → $${((outTok / 1_000_000) * PRICE_OUTPUT_PER_M).toFixed(6)}
          tools:  ${tools} calls
          total:  $${((inTok / 1_000_000) * PRICE_INPUT_PER_M + (outTok / 1_000_000) * PRICE_OUTPUT_PER_M).toFixed(6)}`);

        // type out the response for that "live" feel
        await typeOut(responseBody, outputText);
        conversationHistory.push({ role: 'assistant', content: outputText });

    } catch (err) {
        responseBody.textContent = '[error] ' + err.message;
    }

    // step 4: done
    renderSteps(3);
    clearInterval(elapsedTimer);

    // final metrics: combined input + output
    const totalChars  = inputChars + outputText.length;
    const totalTokens = inTok + outTok;
    const cost        = (inTok  / 1_000_000) * PRICE_INPUT_PER_M
                      + (outTok / 1_000_000) * PRICE_OUTPUT_PER_M;
      
    elChars.textContent  = totalChars;
    elTokens.textContent = totalTokens;
    elCost.textContent   = '$' + cost.toFixed(6);

    isRunning = false;
    runBtn.disabled = false;
    promptInput.disabled = false;
    promptInput.focus();

    // after a moment, mark all steps as done (subdued)
    setTimeout(() => {
      const steps = elSteps.querySelectorAll('.step');
      steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    }, 800);
}

async function typeOut(el, text) {
    el.innerHTML = '';
    const cursor = document.createElement('span');

    cursor.className = 'cursor';
    el.appendChild(cursor);

    const chunkSize = Math.max(1, Math.ceil(text.length / 80));
    for (let i = 0; i < text.length; i += chunkSize) {
      const chunk = text.slice(i, i + chunkSize);
      cursor.insertAdjacentText('beforebegin', chunk);
      transcript.scrollTop = transcript.scrollHeight;
      await sleep(15);
    }

    cursor.remove();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

runBtn.addEventListener('click', execute);
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') execute();
});

renderSteps(-1);
resetSteps();