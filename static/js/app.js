// VoiceFraudShield AI Main Frontend Application Logic

let audioContext = null;
let mediaStream = null;
let audioProcessor = null;
let websocket = null;
let waveformVis = null;
let isStreaming = false;
let currentCallId = "CALL_DEMO";
let modalDismissed = false;

document.addEventListener('DOMContentLoaded', () => {
    initRadarChart();
    waveformVis = new WaveformVisualizer('waveform-canvas');

    loadSimulatorScenarios();
    loadSampleSuite();

    // Register drag-and-drop listeners on upload dropzone
    const dropzone = document.querySelector('.upload-dropzone');
    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = '#00e699';
            dropzone.style.background = 'rgba(0, 230, 153, 0.08)';
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = '';
            dropzone.style.background = '';
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = '';
            dropzone.style.background = '';
            handleFileUpload(e);
        });
    }
});

// Main Feature Navigation Menu Switcher
function switchMainView(viewName) {
    // Update active nav button
    document.querySelectorAll('.nav-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.nav-tab-btn')).find(b => 
        b.getAttribute('onclick').includes(`'${viewName}'`)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const targetPanel = document.getElementById(`view-${viewName}`);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }
}

// Sub-Tab Switcher in Audio Channel Panel
function switchInputSubTab(subTabName) {
    document.querySelectorAll('.sub-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.sub-tab-btn')).find(b => 
        b.getAttribute('onclick').includes(`'${subTabName}'`)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const targetSubTab = document.getElementById(`subtab-${subTabName}`);
    if (targetSubTab) targetSubTab.classList.add('active');

    if (subTabName === 'mic') {
        document.getElementById('transcript-input').value = "";
        document.getElementById('tx-amount-input').value = "0";
        document.getElementById('claimed-speaker-select').value = "";
    }
}

// Attack Simulator Lab Loader
async function loadSimulatorScenarios() {
    try {
        const res = await fetch('/api/v1/simulator/scenarios');
        const data = await res.json();

        const container = document.getElementById('sim-scenarios-container');
        if (!container) return;
        container.innerHTML = '';

        data.scenarios.forEach(scen => {
            const card = document.createElement('div');
            card.className = 'sim-card';
            card.innerHTML = `
                <div class="sim-card-header">
                    <span class="sim-title">${scen.title}</span>
                    <button class="btn btn-danger btn-sm" onclick="runSimulatorScenario('${scen.id}')">
                        <i class="fa-solid fa-bolt"></i> Run Attack Demo
                    </button>
                </div>
                <p class="sim-desc">${scen.description}</p>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        console.error("Failed to load simulator scenarios:", err);
    }
}

async function runSimulatorScenario(scenarioId) {
    try {
        const res = await fetch(`/api/v1/simulator/run/${scenarioId}`, { method: 'POST' });
        const data = await res.json();

        const scen = data.scenario_info;
        if (scen) {
            if (scen.claimed_speaker_id) {
                document.getElementById('claimed-speaker-select').value = scen.claimed_speaker_id;
            }
            document.getElementById('tx-amount-input').value = scen.transaction_amount || 0;
            document.getElementById('transcript-input').value = scen.transcript || '';
        }

        modalDismissed = false;
        updateDashboard(data);

        // Switch to Command Center view so operator instantly sees the attack impact
        switchMainView('command');

        // Play scenario audio
        if (scen && scen.sample_file) {
            const sampleName = scen.sample_file.split('/').pop().replace('.wav','');
            const audio = new Audio(`/api/v1/samples/${sampleName}/audio`);
            audio.play().catch(e => console.log("Audio autoplay prevented"));
        }
    } catch (err) {
        alert("Failed to run scenario: " + err.message);
    }
}

// Step-Up Verification Trigger
async function triggerStepUp(method) {
    try {
        const res = await fetch('/api/v1/verification/request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                call_id: currentCallId,
                method: method
            })
        });

        const data = await res.json();
        alert(`🔒 STEP-UP VERIFICATION INITIATED:\n\n${data.confirmation_message}`);
        closeAlertModal();
    } catch (err) {
        alert("Step-up verification request failed: " + err.message);
    }
}

let speechRecognizer = null;

function getSelectedASRLang() {
    const el = document.getElementById('asr-language-select') || document.getElementById('asr-language-select-upload');
    return el ? el.value : 'auto';
}

function initLiveSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.log("Web Speech API not available; relying on backend Whisper ASR.");
        return null;
    }
    try {
        const recognizer = new SpeechRecognition();
        recognizer.continuous = true;
        recognizer.interimResults = true;
        
        const sel = getSelectedASRLang();
        if (sel === 'hi') {
            recognizer.lang = 'hi-IN';
        } else if (sel === 'mr') {
            recognizer.lang = 'mr-IN';
        } else if (sel === 'en') {
            recognizer.lang = 'en-US';
        } else {
            recognizer.lang = 'hi-IN'; // Auto-detect mode defaults to Hindi/Indian phonetic recognizer
        }

        recognizer.onresult = (event) => {
            let fullText = '';
            for (let i = 0; i < event.results.length; ++i) {
                fullText += event.results[i][0].transcript + ' ';
            }
            fullText = fullText.trim();
            if (fullText) {
                document.querySelectorAll('#transcript-input, #transcript-input-upload, .transcript-display-input').forEach(el => {
                    el.value = fullText;
                });
            }
        };

        recognizer.onerror = (e) => {
            console.log("Speech recognition notification:", e.error);
        };

        return recognizer;
    } catch (e) {
        return null;
    }
}

document.addEventListener('change', (e) => {
    if (e.target && e.target.classList.contains('asr-lang-picker')) {
        const val = e.target.value;
        document.querySelectorAll('.asr-lang-picker').forEach(el => {
            el.value = val;
        });
        if (speechRecognizer) {
            speechRecognizer.lang = (val === 'hi') ? 'hi-IN' : ((val === 'mr') ? 'mr-IN' : ((val === 'en') ? 'en-US' : 'hi-IN'));
        }
    }
});

let modalShownForCallId = null;

// Live Microphone Streaming
async function startMicStream() {
    try {
        currentCallId = "CALL_" + Math.floor(100000 + Math.random() * 900000);
        modalDismissed = false;
        modalShownForCallId = null;

        // Reset all context fields to clean authentic baseline
        document.querySelectorAll('#transcript-input, [name="transcript-input"]').forEach(el => { el.value = ""; });
        if (document.getElementById('tx-amount-input')) document.getElementById('tx-amount-input').value = "0";
        if (document.getElementById('claimed-speaker-select')) document.getElementById('claimed-speaker-select').value = "";
        if (document.getElementById('stir-shaken-select')) document.getElementById('stir-shaken-select').value = "A";
        if (document.getElementById('voip-proxy-select')) document.getElementById('voip-proxy-select').value = "false";
        if (document.getElementById('user-agent-input')) document.getElementById('user-agent-input').value = "Mozilla/5.0 WebRTC/1.0 (Carrier Trunk)";
        if (document.getElementById('cli-number-input')) document.getElementById('cli-number-input').value = "+1-800-555-0199";

        // Start Live Speech Recognition (ASR)
        if (!speechRecognizer) {
            speechRecognizer = initLiveSpeechRecognition();
        }
        if (speechRecognizer) {
            try { speechRecognizer.start(); } catch (e) {}
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

        const source = audioContext.createMediaStreamSource(mediaStream);
        audioProcessor = audioContext.createScriptProcessor(2048, 1, 1);

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream-detect`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log("WebSocket connected with Whisper ASR active. Session:", currentCallId);
            document.getElementById('btn-start-mic').disabled = true;
            document.getElementById('btn-stop-mic').disabled = false;
            isStreaming = true;
            waveformVis.start();
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        audioProcessor.onaudioprocess = (e) => {
            if (!isStreaming || !websocket || websocket.readyState !== WebSocket.OPEN) return;

            const inputData = e.inputBuffer.getChannelData(0);
            waveformVis.updateData(inputData);

            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
            }

            const binaryStr = String.fromCharCode.apply(null, new Uint8Array(pcm16.buffer));
            const base64Audio = btoa(binaryStr);

            const payload = {
                audio_chunk: base64Audio,
                sample_rate: audioContext ? audioContext.sampleRate : 16000,
                context: getContextData()
            };

            websocket.send(JSON.stringify(payload));
        };

        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);

    } catch (err) {
        alert("Microphone access error: " + err.message);
    }
}

function stopMicStream() {
    isStreaming = false;

    if (speechRecognizer) {
        try { speechRecognizer.stop(); } catch (e) {}
        speechRecognizer = null;
    }

    if (websocket) { websocket.close(); websocket = null; }
    if (audioProcessor) { audioProcessor.disconnect(); audioProcessor = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }

    if (waveformVis) waveformVis.stop();

    document.getElementById('btn-start-mic').disabled = false;
    document.getElementById('btn-stop-mic').disabled = true;
}

function getContextData() {
    const transcriptInput = document.getElementById('transcript-input');
    return {
        call_id: currentCallId,
        claimed_speaker_id: document.getElementById('claimed-speaker-select') ? document.getElementById('claimed-speaker-select').value || null : null,
        transaction_amount: document.getElementById('tx-amount-input') ? parseFloat(document.getElementById('tx-amount-input').value) || 0.0 : 0.0,
        transcript: transcriptInput ? transcriptInput.value || "" : "",
        language: getSelectedASRLang(),
        stir_shaken_attestation: document.getElementById('stir-shaken-select') ? document.getElementById('stir-shaken-select').value : "A",
        voip_proxy_detected: document.getElementById('voip-proxy-select') ? (document.getElementById('voip-proxy-select').value === "true") : false,
        user_agent: document.getElementById('user-agent-input') ? document.getElementById('user-agent-input').value : "Mozilla/5.0 WebRTC/1.0",
        cli_number: document.getElementById('cli-number-input') ? document.getElementById('cli-number-input').value : "+1-800-555-0199"
    };
}

let cachedSamples = [];

// Sample Suite Loader & Analyzer
async function loadSampleSuite() {
    try {
        const res = await fetch('/api/v1/samples');
        const data = await res.json();
        cachedSamples = data.samples || [];
        const container = document.getElementById('sample-cards-container');
        if (!container) return;
        container.innerHTML = '';

        cachedSamples.forEach(sample => {
            const isClone = sample.type === 'SYNTHETIC_CLONE';
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.innerHTML = `
                <div class="sample-info">
                    <div class="sample-title">${sample.title}</div>
                    <div class="sample-meta">
                        <span class="${isClone ? 'tag-clone' : 'tag-genuine'}">${sample.type}</span> • ${sample.language}
                    </div>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="analyzeSample('${sample.id}')">
                    <i class="fa-solid fa-play"></i> Test Sample
                </button>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        console.error("Failed to load samples catalog:", err);
    }
}

async function analyzeSample(sampleId) {
    try {
        const sampleMeta = cachedSamples.find(s => s.id === sampleId) || {};
        if (sampleMeta.claimed_speaker_id && document.getElementById('claimed-speaker-select')) {
            document.getElementById('claimed-speaker-select').value = sampleMeta.claimed_speaker_id;
        }
        if (sampleMeta.transaction_amount && document.getElementById('tx-amount-input')) {
            document.getElementById('tx-amount-input').value = sampleMeta.transaction_amount;
        }
        if (sampleMeta.transcript && document.getElementById('transcript-input')) {
            document.getElementById('transcript-input').value = sampleMeta.transcript;
        }

        const audioRes = await fetch(`/api/v1/samples/${sampleId}/audio`);
        const blob = await audioRes.blob();

        const reader = new FileReader();
        reader.onloadend = async () => {
            const base64data = reader.result;

            const payload = {
                audio_base64: base64data,
                claimed_speaker_id: sampleMeta.claimed_speaker_id || (document.getElementById('claimed-speaker-select') ? document.getElementById('claimed-speaker-select').value || null : null),
                transaction_amount: sampleMeta.transaction_amount || (document.getElementById('tx-amount-input') ? parseFloat(document.getElementById('tx-amount-input').value) || 0.0 : 0.0),
                transcript: sampleMeta.transcript || (document.getElementById('transcript-input') ? document.getElementById('transcript-input').value || "" : ""),
                stir_shaken_attestation: document.getElementById('stir-shaken-select') ? document.getElementById('stir-shaken-select').value : "A",
                voip_proxy_detected: document.getElementById('voip-proxy-select') ? (document.getElementById('voip-proxy-select').value === "true") : false
            };

            const analyzeRes = await fetch('/api/v1/analyze-base64', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await analyzeRes.json();
            modalDismissed = false;
            updateDashboard(result);
            switchMainView('command');

            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play().catch(e => console.log("Audio autoplay prevented"));
        };

        reader.readAsDataURL(blob);
    } catch (err) {
        alert("Failed to analyze audio sample: " + err.message);
    }
}

async function handleFileUpload(event) {
    const file = (event.target && event.target.files) ? event.target.files[0] : (event.dataTransfer ? event.dataTransfer.files[0] : null);
    if (!file) return;

    const fileInfo = document.getElementById('file-info');
    if (fileInfo) {
        fileInfo.style.display = 'block';
        fileInfo.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Audio File: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)...`;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('claimed_speaker_id', document.getElementById('claimed-speaker-select') ? document.getElementById('claimed-speaker-select').value || '' : '');
    formData.append('transaction_amount', document.getElementById('tx-amount-input') ? document.getElementById('tx-amount-input').value || '0' : '0');
    formData.append('transcript', document.getElementById('transcript-input') ? document.getElementById('transcript-input').value || '' : '');
    formData.append('language', getSelectedASRLang());
    formData.append('stir_shaken_attestation', document.getElementById('stir-shaken-select') ? document.getElementById('stir-shaken-select').value : 'A');
    formData.append('voip_proxy_detected', document.getElementById('voip-proxy-select') ? (document.getElementById('voip-proxy-select').value === 'true') : false);
    formData.append('user_agent', document.getElementById('user-agent-input') ? document.getElementById('user-agent-input').value : 'Mozilla/5.0 WebRTC/1.0');
    formData.append('cli_number', document.getElementById('cli-number-input') ? document.getElementById('cli-number-input').value : '+1-800-555-0199');

    try {
        const res = await fetch('/api/v1/analyze', { method: 'POST', body: formData });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        modalDismissed = false;
        modalShownForCallId = null;
        updateDashboard(data);
        switchMainView('command');

        const score = (data.risk_summary && data.risk_summary.overall_risk_score) ? data.risk_summary.overall_risk_score.toFixed(1) : 0;
        const tier = (data.risk_summary && data.risk_summary.risk_tier) ? data.risk_summary.risk_tier : 'GREEN';
        if (fileInfo) {
            fileInfo.innerHTML = `<i class="fa-solid fa-circle-check text-green"></i> <strong>${file.name}</strong> analyzed: <span style="font-weight:700;">${tier} RISK (${score}%)</span>`;
        }

        // Play uploaded audio
        try {
            const audioUrl = URL.createObjectURL(file);
            const audio = new Audio(audioUrl);
            audio.play().catch(() => {});
        } catch (e) {}

    } catch (err) {
        if (fileInfo) {
            fileInfo.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red"></i> Error analyzing file: ${err.message}`;
        }
        alert("Failed to analyze audio file: " + err.message);
    } finally {
        if (event.target && event.target.value) {
            event.target.value = '';
        }
    }
}

// Update Dashboard UI with Multi-Vector Analysis Results Across All Views
function updateDashboard(data) {
    currentCallId = data.call_id || "CALL_DEMO";
    const risk = data.risk_summary || {};
    const score = risk.overall_risk_score || 0.0;
    const tier = risk.risk_tier || "GREEN";
    const bd = risk.metric_breakdown || {};
    const net = data.network_gate || {};

    // 1. Gauge & Status Badge in Command Center
    if (document.getElementById('risk-score-num')) {
        document.getElementById('risk-score-num').innerText = score.toFixed(1);
    }

    const meterFill = document.getElementById('meter-fill');
    if (meterFill) {
        const offset = 141.4 - (141.4 * (score / 100));
        meterFill.style.strokeDasharray = "141.4";
        meterFill.style.strokeDashoffset = offset;
        meterFill.className = `meter-fill ${tier.toLowerCase()}`;
    }

    const badge = document.getElementById('risk-status-badge');
    if (badge) {
        badge.className = `risk-badge ${tier.toLowerCase()}`;
        badge.innerText = `${tier} / ${risk.action_code || 'ALLOW'}`;
    }

    const titleEl = document.getElementById('risk-verdict-title');
    if (titleEl) {
        titleEl.className = `verdict-title text-${tier.toLowerCase()}`;
        titleEl.innerText = risk.risk_label || "Genuine Voice & Identity Verified";
    }

    const descEl = document.getElementById('risk-verdict-desc');
    if (descEl) {
        descEl.innerText = `Contextual Risk Fusion Index evaluated at ${score}/100. Call Session ID: ${currentCallId}`;
    }

    // 1B. Synchronize Whisper ASR Transcript Output
    if (data.transcription && data.transcription.transcript) {
        document.querySelectorAll('#transcript-input, [name="transcript-input"]').forEach(el => {
            el.value = data.transcription.transcript;
        });
    }

    // 2. Multi-Vector Progress Bars
    updateBar('m-bar-net', 'm-val-net', bd.network_signaling_score || 0);
    updateBar('m-bar-synth', 'm-val-synth', bd.voice_authenticity_score || 0);
    updateBar('m-bar-spk', 'm-val-spk', bd.speaker_match_score || 0);
    updateBar('m-bar-se', 'm-val-se', bd.social_engineering_score || 0);
    updateBar('m-bar-tx', 'm-val-tx', bd.transaction_context_risk || 0);
    updateBar('m-bar-replay', 'm-val-replay', bd.replay_risk_score || 0);

    // 3. Explainability Reasons List
    const explainList = document.getElementById('explainability-reasons-list');
    if (explainList) {
        explainList.innerHTML = '';
        const exp = risk.explainability || {};
        (exp.reasons || []).forEach(reason => {
            const li = document.createElement('li');
            const isDanger = tier === 'RED' || tier === 'RED_HIGH';
            li.innerHTML = `<i class="fa-solid ${isDanger ? 'fa-triangle-exclamation text-red' : 'fa-check text-green'}"></i> ${reason}`;
            explainList.appendChild(li);
        });
    }

    // 4. Layer 0 Network Gate Telemetry
    if (document.getElementById('stat-em-risk')) {
        document.getElementById('stat-em-risk').innerText = `${net.network_risk_score || 0.05} / 1.0`;
        document.getElementById('stat-em-phase').innerText = net.early_media_details && net.early_media_details.phase_anomaly ? 'VOCODER ARTIFACT' : 'CLEAN';
        document.getElementById('stat-em-phase').className = `stat-val ${net.early_media_details && net.early_media_details.phase_anomaly ? 'text-red' : 'text-green'}`;
        document.getElementById('stat-em-action').innerText = net.action || 'ALLOW 2-WAY';
        document.getElementById('stat-em-action').className = `stat-val ${net.blocked ? 'text-red' : 'text-green'}`;
    }

    // 5. Core Banking Gateway Telemetry (CBS)
    const bank = data.banking_telemetry || {};
    if (bank.transaction_id && document.getElementById('cbs-source-acc')) {
        document.getElementById('cbs-source-acc').innerText = `${bank.source_account} (${bank.source_holder || 'Debtor Account'})`;
        document.getElementById('cbs-target-acc').innerText = `${bank.target_account} (${bank.target_holder || 'Mule Beneficiary'})`;
        document.getElementById('cbs-amount-val').innerText = `₹ ${Number(bank.transaction_amount || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        
        const cbsBadge = document.getElementById('cbs-status-badge');
        const btnBlock = document.getElementById('btn-block-txn');

        if (bank.is_voice_cloned_call || tier === 'RED' || tier === 'RED_HIGH') {
            if (cbsBadge) {
                cbsBadge.className = 'score-tag tag-pending';
                cbsBadge.innerText = 'PENDING / CLONE THREAT DETECTED';
            }
            if (document.getElementById('cbs-latency-val')) {
                document.getElementById('cbs-latency-val').innerText = '⚡ ~38.4 ms (Ready to Freeze)';
            }
            if (btnBlock) {
                btnBlock.disabled = false;
                btnBlock.innerHTML = '<i class="fa-solid fa-lock"></i> Freeze & Block Bank Transaction';
            }
        } else {
            if (cbsBadge) {
                cbsBadge.className = 'score-tag tag-standby';
                cbsBadge.innerText = 'STANDBY / AUTH CLEAN';
            }
            if (document.getElementById('cbs-latency-val')) {
                document.getElementById('cbs-latency-val').innerText = 'Normal (No threat detected)';
            }
        }

        // Modal Banking Callout Elements
        if (document.getElementById('modal-source-acc')) {
            document.getElementById('modal-source-acc').innerText = `${bank.source_account} (${bank.source_holder || 'Debtor'})`;
            document.getElementById('modal-target-acc').innerText = `${bank.target_account} (${bank.target_holder || 'Mule Account'})`;
            document.getElementById('modal-amount-val').innerText = `₹ ${Number(bank.transaction_amount || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        }
    }

    // 6. Social Engineering Tactic Tags
    const tagsBox = document.getElementById('se-tactic-tags-box');
    const se = data.social_engineering_nlp || {};
    if (tagsBox) {
        if (se.tactic_tags && se.tactic_tags.length > 0) {
            if (document.getElementById('se-tactics-count')) {
                document.getElementById('se-tactics-count').innerText = `${se.tactic_tags.length} Detected`;
            }
            tagsBox.innerHTML = se.tactic_tags.map(t => `<span class="tag-tactic">${t}</span>`).join('');
        } else {
            if (document.getElementById('se-tactics-count')) {
                document.getElementById('se-tactics-count').innerText = "0 Detected";
            }
            tagsBox.innerHTML = '<span class="tag-none">No manipulation tactics detected.</span>';
        }
    }

    // 7. Layer 1 Acoustic Diagnostics
    const spec = data.spectral_analysis || {};
    const pros = data.prosody_analysis || {};
    if (document.getElementById('spec-score-val')) {
        document.getElementById('spec-score-val').innerText = `Score: ${spec.synthetic_spectral_score || 0}`;
        document.getElementById('val-flatness').innerText = spec.spectral_flatness || 0;
        document.getElementById('val-hfratio').innerText = spec.high_freq_ratio || 0;
        document.getElementById('val-centroid').innerText = `${spec.spectral_centroid || 0} Hz`;
        document.getElementById('val-jitter-shimmer').innerText = `${pros.jitter || 0} / ${pros.shimmer || 0}`;
    }

    // 8. Deep ML Radar Chart
    const deepMl = data.deep_ml_classification || {};
    if (typeof updateRadarChart === 'function') {
        updateRadarChart(deepMl.feature_attributions || {});
    }

    // 9. Modal Alert Trigger for RED High Risk (Score >= 75) - Exactly ONCE per call
    if ((tier === 'RED' || score >= 75.0) && !modalDismissed && modalShownForCallId !== currentCallId) {
        modalShownForCallId = currentCallId;
        const exp = risk.explainability || {};
        if (document.getElementById('modal-risk-score')) {
            document.getElementById('modal-risk-score').innerText = score.toFixed(1);
        }
        if (document.getElementById('modal-alert-desc')) {
            document.getElementById('modal-alert-desc').innerText = 
                `Contextual Risk Fusion has detected severe threat indicators: ${exp.reasons ? exp.reasons.join(', ') : 'AI voice clone impersonation'}`;
        }
        const modal = document.getElementById('alert-modal');
        if (modal) modal.classList.add('active');
    }
}

function updateBar(barId, valId, val) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(valId);
    if (!bar || !text) return;

    val = Math.min(Math.max(val, 0), 100);
    text.innerText = `${val.toFixed(1)}%`;
    bar.style.width = `${val}%`;

    if (val >= 70) bar.className = 'bar-fill red';
    else if (val >= 40) bar.className = 'bar-fill amber';
    else bar.className = 'bar-fill green';
}

function closeAlertModal() {
    modalDismissed = true;
    modalShownForCallId = currentCallId;
    const modal = document.getElementById('alert-modal');
    if (modal) modal.classList.remove('active');
}

// Execute Core Banking System (CBS) Instant Transaction Freeze
async function executeFreezeTransaction() {
    try {
        const amountVal = document.getElementById('tx-amount-input') ? parseFloat(document.getElementById('tx-amount-input').value) || 3000000.0 : 3000000.0;
        const payload = {
            txn_id: "TXN_NEFT_948201",
            call_id: currentCallId,
            source_account: "ACC-9823418821",
            target_account: "ACC-7729104812",
            amount: amountVal,
            reason: "AI Voice Cloning Impersonation Attack Intercepted by VoiceFraudShield"
        };

        const res = await fetch('/api/v1/banking/transactions/freeze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        // Update UI Badge & Latency
        const badge = document.getElementById('cbs-status-badge');
        if (badge) {
            badge.className = 'score-tag tag-frozen';
            badge.innerText = `FROZEN IN ${data.blocking_latency_ms} ms (LEDGER LOCKED)`;
        }

        if (document.getElementById('cbs-latency-val')) {
            document.getElementById('cbs-latency-val').innerHTML = `<span style="color:#00e699; font-weight:700;">⚡ ${data.blocking_latency_ms} ms</span> (Pre-Clearing Lock Enforced)`;
        }

        const btnBlock = document.getElementById('btn-block-txn');
        if (btnBlock) {
            btnBlock.disabled = true;
            btnBlock.innerHTML = `<i class="fa-solid fa-circle-check"></i> Transaction Locked (${data.blocking_latency_ms}ms)`;
        }

        alert(`🔒 CORE BANKING SYSTEM (CBS) TRANSACTION FROZEN!\n\n` +
              `• Status: AUTOMATICALLY FROZEN / HELD\n` +
              `• Debtor Account: ${data.source_account} (${data.source_holder})\n` +
              `• Beneficiary Account: ${data.target_account} (${data.target_holder})\n` +
              `• Amount: ₹${Number(data.amount).toLocaleString('en-IN', {minimumFractionDigits: 2})}\n` +
              `• Interception Latency: ${data.blocking_latency_ms} ms\n` +
              `• Blocked Timestamp: ${data.blocked_at}\n` +
              `• Audit Reason: ${data.freeze_reason}`);

        closeAlertModal();
    } catch (err) {
        alert("Failed to freeze banking transaction: " + err.message);
    }
}

function triggerBlockAction() {
    executeFreezeTransaction();
}
