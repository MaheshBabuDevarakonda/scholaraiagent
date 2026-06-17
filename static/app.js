document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('evaluationForm');
    const btnSubmit = document.getElementById('btnSubmit');
    const spinner = btnSubmit.querySelector('.spinner');
    const btnText = btnSubmit.querySelector('.btn-text');
    
    const placeholderState = document.getElementById('placeholderState');
    const resultsContent = document.getElementById('resultsContent');
    
    // Result elements
    const lblScore = document.getElementById('lblScore');
    const scoreRing = document.getElementById('scoreRing');
    const lblStatus = document.getElementById('lblStatus');
    const lblEngineMode = document.getElementById('lblEngineMode');
    
    const lblAcademicAnalysis = document.getElementById('lblAcademicAnalysis');
    const lblEnglishAnalysis = document.getElementById('lblEnglishAnalysis');
    const lblFinancialAnalysis = document.getElementById('lblFinancialAnalysis');
    const lblGapAnalysis = document.getElementById('lblGapAnalysis');
    
    const lstUniversities = document.getElementById('lstUniversities');
    const lstRecommendations = document.getElementById('lstRecommendations');

    // SVG Circle properties for score ring
    const radius = scoreRing.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    
    // Set up stroke dash arrays
    scoreRing.style.strokeDasharray = `${circumference} ${circumference}`;
    scoreRing.style.strokeDashoffset = circumference;

    function setScore(score, status) {
        // Animate count up
        let start = 0;
        const duration = 1000; // ms
        const startTime = performance.now();

        function animateCount(timestamp) {
            const runtime = timestamp - startTime;
            const progress = Math.min(runtime / duration, 1);
            const currentScore = Math.floor(progress * score);
            lblScore.textContent = currentScore;
            
            // Offset logic: offset = circumference - (pct * circumference)
            const offset = circumference - (progress * (score / 100) * circumference);
            scoreRing.style.strokeDashoffset = offset;

            if (runtime < duration) {
                requestAnimationFrame(animateCount);
            } else {
                lblScore.textContent = score;
                scoreRing.style.strokeDashoffset = circumference - ((score / 100) * circumference);
            }
        }
        
        // Dynamic ring color based on status
        if (status === 'Green') {
            scoreRing.style.stroke = '#10b981'; // Success Green
        } else if (status === 'Yellow') {
            scoreRing.style.stroke = '#f59e0b'; // Warning Yellow
        } else {
            scoreRing.style.stroke = '#ef4444'; // Danger Red
        }

        requestAnimationFrame(animateCount);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get values
        const profile = {
            destination: document.getElementById('destination').value,
            gpa: parseFloat(document.getElementById('gpa').value),
            gpa_scale: document.getElementById('gpa_scale').value,
            english_test: document.getElementById('english_test').value,
            english_score: parseFloat(document.getElementById('english_score').value) || 0,
            budget_lakhs: parseFloat(document.getElementById('budget').value),
            gap_years: parseInt(document.getElementById('gap_years').value) || 0,
            work_exp_years: parseInt(document.getElementById('work_exp').value) || 0,
            backlogs: parseInt(document.getElementById('backlogs').value) || 0
        };

        // UI Loading state
        btnSubmit.disabled = true;
        spinner.classList.remove('hidden');
        btnText.textContent = "Analyzing Profile...";

        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(profile)
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            const data = await response.json();

            // Transition Panels
            placeholderState.classList.add('hidden');
            resultsContent.classList.remove('hidden');

            // Set Engine/Model Indicator Status
            if (data.is_mock) {
                lblEngineMode.textContent = "Evaluated via Local Policy Engine (Mock Mode)";
                lblEngineMode.style.color = "#f59e0b";
            } else {
                lblEngineMode.textContent = "AI Evaluated via Gemini 2.5";
                lblEngineMode.style.color = "#94a3b8";
            }

            // Status Badge styling
            lblStatus.className = 'badge-status'; // reset
            lblStatus.textContent = data.status.toUpperCase();
            if (data.status === 'Green') {
                lblStatus.classList.add('status-green');
            } else if (data.status === 'Yellow') {
                lblStatus.classList.add('status-yellow');
            } else {
                lblStatus.classList.add('status-red');
            }

            // Core Analyses
            lblAcademicAnalysis.textContent = data.academic_analysis;
            lblEnglishAnalysis.textContent = data.english_analysis;
            lblFinancialAnalysis.textContent = data.financial_analysis;
            lblGapAnalysis.textContent = data.gap_analysis;

            // Render matched universities
            lstUniversities.innerHTML = '';
            if (data.matched_universities && data.matched_universities.length > 0) {
                data.matched_universities.forEach(uni => {
                    const div = document.createElement('div');
                    div.className = 'uni-item';
                    div.innerHTML = `
                        <h5>${uni.name}</h5>
                        <div class="uni-course">${uni.course}</div>
                        <p class="uni-rationale">${uni.rationale}</p>
                    `;
                    lstUniversities.appendChild(div);
                });
            } else {
                lstUniversities.innerHTML = '<div class="uni-rationale">No suitable matches found for this profile. Try adjusting inputs.</div>';
            }

            // Render recommendations/warnings
            lstRecommendations.innerHTML = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    lstRecommendations.appendChild(li);
                });
            } else {
                lstRecommendations.innerHTML = '<li>Profile meets key visa guidelines. Standard documents required.</li>';
            }

            // Animate Score circular progress
            setScore(data.score, data.status);

        } catch (error) {
            console.error("Evaluation error:", error);
            alert("An error occurred during evaluation. Please verify server connection.");
        } finally {
            // Restore button
            btnSubmit.disabled = false;
            spinner.classList.add('hidden');
            btnText.textContent = "Evaluate Profile";
        }
    });

    // Disable English score input if "None" test selected
    const englishTestDropdown = document.getElementById('english_test');
    const englishScoreInput = document.getElementById('english_score');

    englishTestDropdown.addEventListener('change', () => {
        if (englishTestDropdown.value === 'None') {
            englishScoreInput.value = '0';
            englishScoreInput.disabled = true;
            englishScoreInput.required = false;
        } else {
            englishScoreInput.disabled = false;
            englishScoreInput.required = true;
            englishScoreInput.value = '';
        }
    });
});
