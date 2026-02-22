// ClimateGuard Frontend Application

const API_URL = window.location.origin; // use same origin as served frontend
let allRegions = [];

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('ClimateGuard initializing...');
    loadRegions();
    loadAlerts();
    updateDashboardStats();
    initInteractions();
});

// Tab Navigation
function showTab(tabName, evt) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Activate corresponding button (supports optional event)
    const clicked = evt?.target || document.querySelector(`.nav-btn[onclick*="${tabName}"]`);
    if (clicked) clicked.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'alerts') {
        loadAlerts();
    } else if (tabName === 'regions') {
        displayRegions();
    }
}

// Load all regions
async function loadRegions() {
    try {
        const response = await fetch(`${API_URL}/regions`);
        const data = await response.json();
        allRegions = data.regions;
        
        // Update total regions count
        document.getElementById('total-regions').textContent = allRegions.length;
        
        // Populate region checkboxes
        const checkboxContainer = document.getElementById('region-checkboxes');
        checkboxContainer.innerHTML = '';
        
        allRegions.forEach(region => {
            const label = document.createElement('label');
            label.className = 'checkbox-label';
            label.innerHTML = `
                <input type="checkbox" value="${region.id}" checked>
                ${region.name}
            `;
            checkboxContainer.appendChild(label);
        });

        // Populate region select dropdown (if present)
        const regionSelect = document.getElementById('region-select');
        if (regionSelect) {
            // keep 'all' and 'custom' options and add regions in between
            // remove existing region-specific options except 'all' and 'custom'
            const keep = ['all', 'custom'];
            // remove all non-keep except leave order intact
            Array.from(regionSelect.options).forEach(opt => {
                if (!keep.includes(opt.value)) regionSelect.removeChild(opt);
            });

            // Insert region options after 'all' option
            const allOpt = regionSelect.querySelector('option[value="all"]');
            let insertAfter = allOpt || null;
            allRegions.forEach(region => {
                const opt = document.createElement('option');
                opt.value = String(region.id);
                opt.textContent = region.name;
                if (insertAfter && insertAfter.nextSibling) {
                    regionSelect.insertBefore(opt, insertAfter.nextSibling);
                    insertAfter = opt;
                } else {
                    regionSelect.appendChild(opt);
                    insertAfter = opt;
                }
            });

            // When user changes select, if they choose a specific region, uncheck others and check that one
            regionSelect.addEventListener('change', (e) => {
                const val = e.target.value;
                const checkboxes = document.querySelectorAll('#region-checkboxes input[type="checkbox"]');
                if (val === 'custom') {
                    // do nothing (manual checkboxes)
                    return;
                } else if (val === 'all') {
                    checkboxes.forEach(cb => cb.checked = true);
                } else {
                    // single region selected by id
                    checkboxes.forEach(cb => cb.checked = (cb.value === val));
                }
            });
        }
        
        console.log(`Loaded ${allRegions.length} regions`);
    } catch (error) {
        console.error('Error loading regions:', error);
        showError('Failed to load regions. Please check if the API server is running.');
    }
}

// Display regions in grid
function displayRegions() {
    const container = document.getElementById('regions-list');
    container.innerHTML = '';
    
    allRegions.forEach(region => {
        const card = document.createElement('div');
        card.className = 'region-card';
        card.setAttribute('data-reveal', 'true');
        card.innerHTML = `
            <h3>📍 ${region.name}</h3>
            <p><strong>ID:</strong> ${region.id}</p>
            <p><strong>Latitude:</strong> ${region.lat.toFixed(4)}</p>
            <p><strong>Longitude:</strong> ${region.lon.toFixed(4)}</p>
        `;
        container.appendChild(card);
    });
}

// Get predictions
async function getPredictions() {
    const disasterType = document.getElementById('disaster-type').value;
    const forecastDays = parseInt(document.getElementById('forecast-days').value);
    
    // Determine selected regions: either from region-select or manual checkboxes
    const regionSelect = document.getElementById('region-select');
    let regionIds = [];
    if (regionSelect && regionSelect.value && regionSelect.value !== 'custom') {
        if (regionSelect.value === 'all') {
            regionIds = allRegions.map(r => r.id);
        } else {
            // single region selected (value is region id)
            regionIds = [parseInt(regionSelect.value)];
        }
    } else {
        // Get selected regions from checkboxes
        const checkboxes = document.querySelectorAll('#region-checkboxes input:checked');
        regionIds = Array.from(checkboxes).map(cb => parseInt(cb.value));
    }
    
    if (regionIds.length === 0) {
        alert('Please select at least one region');
        return;
    }
    
    const resultsContainer = document.getElementById('prediction-results');
    resultsContainer.innerHTML = '<p class="loading">⏳ Generating predictions...</p>';
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                region_ids: regionIds,
                disaster_type: disasterType,
                forecast_days: forecastDays
            })
        });
        
        const data = await response.json();
        displayPredictionResults(data);
        
        // Update dashboard
        updateRecentPredictions(data);
        
    } catch (error) {
        console.error('Error getting predictions:', error);
        resultsContainer.innerHTML = `
            <div class="alert-card">
                <p>❌ Error getting predictions. Please ensure the API server is running at ${API_URL}</p>
            </div>
        `;
    }
}

// Display prediction results
function displayPredictionResults(data) {
    const container = document.getElementById('prediction-results');
    container.innerHTML = '';
    
    data.predictions.forEach(regionPred => {
        const regionDiv = document.createElement('div');
        regionDiv.className = 'region-prediction';
        regionDiv.setAttribute('data-reveal', 'true');
        
        let tableHTML = `
            <h3>${regionPred.region_name}</h3>
            <div class="risk-badge ${regionPred.overall_risk}">${regionPred.overall_risk.toUpperCase()}</div>
            <table class="forecast-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>${data.disaster_type === 'flood' ? 'Flood' : 'Heatwave'} Risk</th>
                        <th>Probability</th>
                        <th>Temperature</th>
                        <th>Rainfall</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        regionPred.forecasts.forEach(forecast => {
            const prob = data.disaster_type === 'flood' ? 
                         forecast.flood_probability : 
                         forecast.heatwave_probability;
            const risk = data.disaster_type === 'flood' ? 
                         forecast.flood_risk : 
                         forecast.heatwave_risk;
            
            tableHTML += `
                <tr>
                    <td>${forecast.date}</td>
                    <td><span class="risk-badge ${risk}">${risk}</span></td>
                    <td>${(prob * 100).toFixed(1)}%</td>
                    <td>${forecast.temperature}°C</td>
                    <td>${forecast.rainfall} mm</td>
                </tr>
            `;
        });
        
        tableHTML += `
                </tbody>
            </table>
        `;
        
        regionDiv.innerHTML = tableHTML;
        container.appendChild(regionDiv);
    });
}

// Update recent predictions on dashboard
function updateRecentPredictions(data) {
    const container = document.getElementById('recent-predictions');
    container.innerHTML = '';
    
    data.predictions.forEach(regionPred => {
        const item = document.createElement('div');
        item.className = `prediction-item risk-${regionPred.overall_risk}`;
        item.setAttribute('data-reveal', 'true');
        item.innerHTML = `
            <h4>${regionPred.region_name}</h4>
            <div class="risk-badge ${regionPred.overall_risk}">${regionPred.overall_risk.toUpperCase()}</div>
            <p><strong>Disaster Type:</strong> ${data.disaster_type}</p>
            <p><strong>Forecast Period:</strong> ${data.forecast_days} days</p>
        `;
        container.appendChild(item);
    });
    
    // Update prediction count
    document.getElementById('predictions-count').textContent = data.predictions.length;
}

// Load active alerts
async function loadAlerts() {
    const container = document.getElementById('alerts-list');
    container.innerHTML = '<p class="loading">⏳ Loading alerts...</p>';
    
    try {
        const response = await fetch(`${API_URL}/alerts`);
        const data = await response.json();
        
        displayAlerts(data.alerts);
        
        // Update dashboard stats
        document.getElementById('active-warnings').textContent = data.alerts.length;
        
    } catch (error) {
        console.error('Error loading alerts:', error);
        container.innerHTML = `
            <div class="alert-card">
                <p>❌ Error loading alerts. Please check if the API server is running.</p>
            </div>
        `;
    }
}

// Display alerts
function displayAlerts(alerts) {
    const container = document.getElementById('alerts-list');
    container.innerHTML = '';
    
    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="alert-card" style="border-left-color: var(--success);">
                <h4>✅ No Active Alerts</h4>
                <p>All monitored regions are currently safe.</p>
            </div>
        `;
        return;
    }
    
    alerts.forEach(alert => {
        const card = document.createElement('div');
        card.className = 'alert-card';
        card.setAttribute('data-reveal', 'true');
        card.innerHTML = `
            <h4>🚨 ${alert.region_name}</h4>
            <div class="alert-severity ${alert.severity}">${alert.severity.toUpperCase()}</div>
            <p><strong>Type:</strong> ${alert.type}</p>
            <p><strong>Probability:</strong> ${(alert.probability * 100).toFixed(1)}%</p>
            <p>${alert.message}</p>
        `;
        container.appendChild(card);
    });
}

/* Interactions: mouse parallax, scroll progress, and reveal */
function initInteractions() {
    // Parallax: move hero layers slightly based on mouse
    const hero = document.querySelector('.hero');
    const layers = document.querySelectorAll('.hero-layer[data-depth]');

    document.addEventListener('mousemove', (e) => {
        if (!layers.length) return;
        const rect = hero.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = (e.clientX - cx) / rect.width;
        const dy = (e.clientY - cy) / rect.height;

        layers.forEach(layer => {
            const depth = parseFloat(layer.dataset.depth) || 0.03;
            const tx = dx * 30 * depth;
            const ty = dy * 30 * depth;
            layer.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
        });
    });

    // Scroll progress
    const progress = document.getElementById('scroll-progress');
    function updateProgress() {
        const h = document.documentElement.scrollHeight - window.innerHeight;
        const pct = h > 0 ? (window.scrollY / h) * 100 : 0;
        progress.style.width = pct + '%';
    }
    document.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();

    // Reveal on scroll using IntersectionObserver
    const io = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                io.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });

    // Observe existing reveal elements
    document.querySelectorAll('[data-reveal]').forEach(el => io.observe(el));

    // Also watch for future elements added dynamically with data-reveal
    const mo = new MutationObserver(mutations => {
        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (!(node instanceof HTMLElement)) return;
                if (node.matches('[data-reveal]')) io.observe(node);
                node.querySelectorAll && node.querySelectorAll('[data-reveal]').forEach(el => io.observe(el));
            });
        });
    });
    mo.observe(document.body, { childList: true, subtree: true });
}

// Update dashboard statistics
async function updateDashboardStats() {
    try {
        const alertsResponse = await fetch(`${API_URL}/alerts`);
        const alertsData = await alertsResponse.json();
        
        // Count warnings and safe zones
        const warnings = alertsData.alerts.length;
        const safeZones = allRegions.length - warnings;
        
        document.getElementById('active-warnings').textContent = warnings;
        document.getElementById('safe-zones').textContent = safeZones;
        
    } catch (error) {
        console.error('Error updating dashboard stats:', error);
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert-card';
    errorDiv.innerHTML = `<p>❌ ${message}</p>`;
    errorDiv.style.position = 'fixed';
    errorDiv.style.top = '20px';
    errorDiv.style.right = '20px';
    errorDiv.style.zIndex = '1000';
    errorDiv.style.background = 'white';
    errorDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Helper function to format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

console.log('ClimateGuard frontend loaded successfully!');
console.log(`API URL: ${API_URL}`);