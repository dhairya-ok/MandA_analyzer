import { useEffect, useState, useCallback } from 'react'
import './styles.css'
import LoadingSpinner from './components/LoadingSpinner'

interface ApiError {
    message: string;
    type?: string;
}

const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000'

export default function App() {
	const [health, setHealth] = useState<string>('checking…')

	const [acquirer, setAcquirer] = useState<string>('')
	const [top, setTop] = useState<number>(10)
	const [pairs, setPairs] = useState<any[]>([])
	const [loadingPairs, setLoadingPairs] = useState<boolean>(false)
	const [generatingBrief, setGeneratingBrief] = useState<string | null>(null)

	useEffect(() => {
		fetch(`${apiBase}/health`).then(r => r.json()).then(j => setHealth(j.status || 'ok')).catch(() => setHealth('offline'))
	}, [])

	function onGetStarted() {
		window.scrollTo({ top: document.body.clientHeight, behavior: 'smooth' })
	}

	function onOpenScreener() {
		// scroll to features which now include screener controls
		window.location.hash = '#screener'
	}

	async function fetchPairs() {
		if (!acquirer) return alert('Enter acquirer ticker (e.g., AAPL)')
		setLoadingPairs(true)
		try {
			const resp = await fetch(`${apiBase}/pairs?acquirer=${encodeURIComponent(acquirer)}&top=${top}`)
			if (!resp.ok) throw new Error(await resp.text())
			const j = await resp.json()
			setPairs(j.results || [])
		} catch (e) {
			console.error(e)
			alert('Failed to fetch pairs: ' + e)
		} finally {
			setLoadingPairs(false)
		}
	}

	return (
		<div>
			<div className="container nav">
				<div className="brand">Mergeriny</div>
				<div>
					<a href="#features">Features</a>
					<a href="#valuation">Valuation</a>
					<a href="#export">Deal Brief</a>
				</div>
			</div>

			<header className="container hero">
				<div className="kicker">M&A screening, valuation, and briefs</div>
				<h1 className="title">Find the perfect acquisition. In minutes, not days.</h1>
				<p className="subtitle">An end‑to‑end M&A platform that discovers targets, scores strategic fit, triangulates value with multi‑model methods, and generates beautiful Deal Briefs — fast.</p>
				<div className="cta-row">
					<button className="btn primary" onClick={onGetStarted}>Get started</button>
					<button className="btn" onClick={onOpenScreener}>Open Screener</button>
				</div>
				<div className="badges">
					<span>API: {health}</span>
					<span>Base URL: {apiBase}</span>
				</div>
			</header>

			<main className="container">
				<section id="features" className="section">
					<div className="grid">
						<div className="card">
							<h3>Deal Generation</h3>
							<p>Sector fit, size ratio, capacity, growth complementarity, and correlation — composited to a 0–100 score.</p>
						</div>
						<div className="card">
							<h3>Valuation Orchestration</h3>
							<p>DCF, Comps, and Rule Engine ensemble. Assumptions are transparent and editable with live sensitivity.</p>
						</div>
						<div className="card">
							<h3>Deal Brief Export</h3>
							<p>Generate a polished PDF brief: strategic rationale, model breakdowns, and an auditable offer range.</p>
						</div>
					</div>
				</section>

				<section id="screener" className="section">
					<div className="card">
						<h3>Screener</h3>
						<p>Enter an acquirer ticker to see top target suggestions generated from the current universe.</p>
						<div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
							<input placeholder="ACQUIRER (e.g., AAPL)" value={acquirer} onChange={e => setAcquirer(e.target.value.toUpperCase())} />
							<input type="number" value={top} onChange={e => setTop(Number(e.target.value || 10))} style={{ width: 80 }} />
							<button className="btn primary" onClick={fetchPairs} disabled={loadingPairs}>{loadingPairs ? 'Loading…' : 'Get Top'}</button>
						</div>
						{pairs.length > 0 && (
							<table style={{ width: '100%', marginTop: 12 }}>
								<thead>
									<tr>
										<th>Rank</th>
										<th>Ticker</th>
										<th>Score</th>
										<th>Actions</th>
										<th>Name</th>
										<th>Score</th>
										<th>Market Cap</th>
									</tr>
								</thead>
								<tbody>
									{pairs.map((p, i) => (
										<tr key={p.target}>
											<td>{i + 1}</td>
											<td>{p.target}</td>
											<td>{p.target_name}</td>
											<td>{p.score.toFixed(1)}</td>
											<td>{p.target_market_cap ? (Number(p.target_market_cap).toLocaleString()) : '—'}</td>
										</tr>
									))}
								</tbody>
							</table>
						)}
					</div>
				</section>

				<section id="valuation" className="section">
					<div className="grid">
						<div className="card">
							<h3>DCF (stub)</h3>
							<p>Call the API DCF endpoint once data ingestion is wired.</p>
						</div>
						<div className="card">
							<h3>Comps (stub)</h3>
							<p>Peer multiples and implied ranges, with confidence based on coverage.</p>
						</div>
						<div className="card">
							<h3>Sensitivity (stub)</h3>
							<p>WACC ±2%, g ±1%, and multiple ±2x heatmaps coming soon.</p>
						</div>
					</div>
				</section>

				<section id="export" className="section">
					<div className="card">
						<h3>One‑click Deal Brief</h3>
						<p>Export a professional PDF for interviews — with provenance and assumptions clearly documented.</p>
					</div>
				</section>
			</main>

			<div className="footer">© {new Date().getFullYear()} Mergeriny — Built for speed and clarity.</div>
		</div>
	)
}
