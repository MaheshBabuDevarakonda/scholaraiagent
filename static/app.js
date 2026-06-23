document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('evaluationForm');
    const btnSubmit = document.getElementById('btnSubmit');
    const btnText = btnSubmit.querySelector('.btn-text');
    const btnLoader = btnSubmit.querySelector('.btn-loader');

    const placeholderState = document.getElementById('placeholderState');
    const resultsContent = document.getElementById('resultsContent');

    // Result elements
    const lblScore = document.getElementById('lblScore');
    const scoreRing = document.getElementById('scoreRing');
    const lblStatus = document.getElementById('lblStatus');
    const lblEngineMode = document.getElementById('lblEngineMode');

    const lblEducationAnalysis = document.getElementById('lblEducationAnalysis');
    const lblEnglishAnalysis = document.getElementById('lblEnglishAnalysis');
    const lblFinancialAnalysis = document.getElementById('lblFinancialAnalysis');
    const lblAgeAnalysis = document.getElementById('lblAgeAnalysis');
    const lblExperienceAnalysis = document.getElementById('lblExperienceAnalysis');

    const lstVisaOptions = document.getElementById('lstVisaOptions');
    const lstRecommendations = document.getElementById('lstRecommendations');

    // SVG ring setup
    const radius = scoreRing.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    scoreRing.style.strokeDasharray = `${circumference} ${circumference}`;
    scoreRing.style.strokeDashoffset = circumference;

    // ── Score Animation ──
    function setScore(score, status) {
        const duration = 1200;
        const startTime = performance.now();

        // Pick gradient
        if (status === 'Eligible') {
            scoreRing.style.stroke = 'url(#ringGradGreen)';
        } else {
            scoreRing.style.stroke = 'url(#ringGradRed)';
        }

        function animate(timestamp) {
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);

            const currentScore = Math.floor(eased * score);
            lblScore.textContent = currentScore;

            const offset = circumference - (eased * (score / 100) * circumference);
            scoreRing.style.strokeDashoffset = offset;

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                lblScore.textContent = score;
                scoreRing.style.strokeDashoffset = circumference - ((score / 100) * circumference);
            }
        }

        requestAnimationFrame(animate);
    }

    // ── Form Submission ──
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const profile = {
            citizenship: document.getElementById('citizenship').value,
            destination: document.getElementById('destination').value,
            age: parseInt(document.getElementById('age').value) || 0,
            education: document.getElementById('education').value,
            work_exp_years: parseInt(document.getElementById('work_exp').value) || 0,
            english_test: document.getElementById('english_test').value,
            english_score: parseFloat(document.getElementById('english_score').value) || 0,
            funds_lakhs: parseFloat(document.getElementById('funds').value) || 0,
            purpose: document.getElementById('purpose').value
        };

        // Loading state
        btnSubmit.disabled = true;
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');

        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile)
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            const data = await response.json();

            // Transition panels
            placeholderState.classList.add('hidden');
            resultsContent.classList.remove('hidden');

            // Re-trigger animations by removing and re-adding the class
            resultsContent.style.animation = 'none';
            resultsContent.offsetHeight; // force reflow
            resultsContent.style.animation = '';

            // Engine indicator
            if (data.is_mock) {
                lblEngineMode.textContent = 'Local Policy Engine (Mock Mode)';
                lblEngineMode.style.color = '#fbbf24';
            } else {
                lblEngineMode.textContent = 'AI Evaluated via Gemini 2.5';
                lblEngineMode.style.color = '#94a3b8';
            }

            // Status badge
            lblStatus.className = 'status-badge';
            lblStatus.textContent = data.status === 'Eligible' ? 'ELIGIBLE' : 'NOT POSSIBLE';
            if (data.status === 'Eligible') {
                lblStatus.classList.add('status-green');
            } else {
                lblStatus.classList.add('status-red');
            }

            // Analysis cards
            lblEducationAnalysis.textContent = data.education_analysis;
            lblEnglishAnalysis.textContent = data.english_analysis;
            lblFinancialAnalysis.textContent = data.financial_analysis;
            lblAgeAnalysis.textContent = data.age_analysis;
            lblExperienceAnalysis.textContent = data.experience_analysis;

            // Visa Options
            lstVisaOptions.innerHTML = '';
            if (data.visa_options && data.visa_options.length > 0) {
                data.visa_options.forEach((opt, i) => {
                    const div = document.createElement('div');
                    div.className = 'uni-item';
                    
                    // Style status badge
                    let badgeClass = 'status-red';
                    if (opt.status === 'Eligible') badgeClass = 'status-green';
                    else if (opt.status === 'Potential') badgeClass = 'status-yellow';
                    
                    div.style.animationDelay = `${0.3 + i * 0.08}s`;
                    div.style.animation = `fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) ${0.3 + i * 0.08}s both`;
                    div.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <h5 style="margin: 0;">${opt.name}</h5>
                            <span class="status-badge ${badgeClass}" style="font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 700;">${opt.status.toUpperCase()}</span>
                        </div>
                        <p class="uni-rationale" style="margin-top: 4px;">${opt.rationale}</p>
                    `;
                    lstVisaOptions.appendChild(div);
                });
            } else {
                lstVisaOptions.innerHTML = '<div class="uni-rationale">No suitable visa pathways found for this profile. Try adjusting inputs.</div>';
            }

            // Recommendations
            lstRecommendations.innerHTML = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach((rec, i) => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    li.style.animation = `fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) ${0.4 + i * 0.06}s both`;
                    lstRecommendations.appendChild(li);
                });
            } else {
                lstRecommendations.innerHTML = '<li>Profile meets key visa guidelines. Standard documents required.</li>';
            }

            // Animate score ring
            setScore(data.score, data.status);

            // Smooth scroll to results on mobile
            if (window.innerWidth <= 1100) {
                document.getElementById('resultsPanel').scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }

        } catch (error) {
            console.error('Evaluation error:', error);

            // Show inline error instead of alert
            placeholderState.classList.add('hidden');
            resultsContent.classList.remove('hidden');
            resultsContent.innerHTML = `
                <div class="empty-state" style="color: var(--red);">
                    <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
                    <h3 style="color: var(--red);">Evaluation Failed</h3>
                    <p style="color: var(--text-muted); margin-top: 8px;">
                        Could not complete the evaluation. Please verify the server is running and try again.
                    </p>
                    <p style="color: var(--text-dim); font-size: 11px; margin-top: 12px; font-family: 'JetBrains Mono', monospace;">
                        ${error.message}
                    </p>
                </div>
            `;
        } finally {
            btnSubmit.disabled = false;
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    });

    // ── English Test Toggle ──
    const englishTestDropdown = document.getElementById('english_test');
    const englishScoreInput = document.getElementById('english_score');

    englishTestDropdown.addEventListener('change', () => {
        if (englishTestDropdown.value === 'None') {
            englishScoreInput.value = '0';
            englishScoreInput.disabled = true;
            englishScoreInput.required = false;
            englishScoreInput.style.opacity = '0.4';
        } else {
            englishScoreInput.disabled = false;
            englishScoreInput.required = true;
            englishScoreInput.value = '';
            englishScoreInput.style.opacity = '1';
        }
    });

    // ── Navbar shadow on scroll ──
    const topNav = document.getElementById('topNav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            topNav.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(99, 102, 241, 0.06)';
        } else {
            topNav.style.boxShadow = 'none';
        }
    }, { passive: true });

    // ── Form input interaction feedback ──
    document.querySelectorAll('.form-group input, .form-group select').forEach(el => {
        el.addEventListener('focus', () => {
            el.closest('.form-group').style.transform = 'translateX(4px)';
            el.closest('.form-group').style.transition = 'transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)';
        });
        el.addEventListener('blur', () => {
            el.closest('.form-group').style.transform = 'translateX(0)';
        });
    });
});
