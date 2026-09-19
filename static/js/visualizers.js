// Audio Visualizers & Charting Module

class WaveformVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.animFrame = null;
        this.audioData = new Float32Array(256);
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = this.canvas.clientWidth * window.devicePixelRatio;
        this.canvas.height = this.canvas.clientHeight * window.devicePixelRatio;
    }

    updateData(dataArray) {
        this.audioData = dataArray;
    }

    start() {
        const render = () => {
            if (!this.ctx) return;
            const w = this.canvas.width;
            const h = this.canvas.height;

            this.ctx.clearRect(0, 0, w, h);

            // Draw center baseline
            this.ctx.beginPath();
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            this.ctx.lineWidth = 1;
            this.ctx.moveTo(0, h / 2);
            this.ctx.lineTo(w, h / 2);
            this.ctx.stroke();

            // Draw oscilloscope waveform
            this.ctx.beginPath();
            this.ctx.strokeStyle = '#00e699';
            this.ctx.lineWidth = 2 * window.devicePixelRatio;
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = '#00e699';

            const sliceWidth = w / this.audioData.length;
            let x = 0;

            for (let i = 0; i < this.audioData.length; i++) {
                const v = this.audioData[i];
                const y = (v * (h / 2.5)) + (h / 2);

                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            this.ctx.stroke();
            this.animFrame = requestAnimationFrame(render);
        };

        render();
    }

    stop() {
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
        }
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
}

// Chart.js Radar Chart
let radarChartInstance = null;

function initRadarChart() {
    const canvas = document.getElementById('artifact-radar-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Spectral Flatness', 'Vocoder HF Artifact', 'Jitter Anomaly', 'Shimmer Anomaly', 'Pitch Jumps'],
            datasets: [{
                label: 'Artifact Risk Attribution',
                data: [0.1, 0.1, 0.1, 0.1, 0.1],
                backgroundColor: 'rgba(255, 51, 102, 0.2)',
                borderColor: '#ff3366',
                borderWidth: 2,
                pointBackgroundColor: '#ff3366',
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { size: 9, family: 'Outfit' }
                    },
                    ticks: { display: false },
                    min: 0,
                    max: 1.0
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateRadarChart(attributions) {
    if (!radarChartInstance) return;

    const data = [
        attributions.spectral_flatness_risk || 0.1,
        attributions.vocoder_hf_artifact_risk || 0.1,
        attributions.jitter_anomaly_risk || 0.1,
        attributions.shimmer_anomaly_risk || 0.1,
        attributions.pitch_jump_risk || 0.1
    ];

    radarChartInstance.data.datasets[0].data = data;
    radarChartInstance.update();
}

window.WaveformVisualizer = WaveformVisualizer;
window.initRadarChart = initRadarChart;
window.updateRadarChart = updateRadarChart;
