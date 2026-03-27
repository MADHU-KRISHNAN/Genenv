import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predict, getSamplePatient, type PatientData, type PredictionResult, type SamplePatient } from '../api/client';

function GaugeChart({ value }: { value: number }) {
    const pct = Math.round(value * 100);
    const circumference = 283;
    const offset = circumference - (circumference * pct) / 100;
    const color = pct < 30 ? '#22c55e' : pct < 60 ? '#f59e0b' : '#ef4444';

    return (
        <div className="relative w-48 h-48 mx-auto">
            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#1e293b" strokeWidth="8" />
                <circle
                    cx="50" cy="50" r="45" fill="none" stroke={color} strokeWidth="8"
                    strokeLinecap="round" strokeDasharray={circumference}
                    strokeDashoffset={offset} className="gauge-circle"
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-4xl font-bold" style={{ color }}>{pct}%</span>
                <span className="text-xs text-slate-500 mt-1">Mortality Risk</span>
            </div>
        </div>
    );
}

function ContributionBar({ feature, contribution }: { feature: string; contribution: number }) {
    const maxWidth = 100;
    const width = Math.min(Math.abs(contribution) * 1000, maxWidth);
    const positive = contribution > 0;
    return (
        <div className="flex items-center gap-3 text-sm mb-2">
            <span className="w-32 truncate text-slate-400 text-right">{feature}</span>
            <div className="flex-1 flex items-center gap-1">
                {!positive && (
                    <div className="h-5 rounded-sm bg-green-500/70" style={{ width: `${width}%`, marginLeft: 'auto' }} />
                )}
                <div className="w-px h-6 bg-slate-600 flex-shrink-0" />
                {positive && (
                    <div className="h-5 rounded-sm bg-red-500/70" style={{ width: `${width}%` }} />
                )}
            </div>
            <span className={`w-16 text-xs ${positive ? 'text-red-400' : 'text-green-400'}`}>
                {contribution > 0 ? '+' : ''}{(contribution * 100).toFixed(2)}%
            </span>
        </div>
    );
}

export default function Predict() {
    const [sample, setSample] = useState<SamplePatient | null>(null);
    const [age, setAge] = useState(65);
    const [gender, setGender] = useState('MALE');
    const [smoking, setSmoking] = useState('former_gt15');
    const [stage, setStage] = useState('II');
    const [geneValues, setGeneValues] = useState<Record<string, number>>({});
    const [methylValues, setMethylValues] = useState<Record<string, number>>({});
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [history, setHistory] = useState<{ date: string; prob: number; risk: string }[]>([]);

    useEffect(() => {
        const saved = localStorage.getItem('gxe_history');
        if (saved) setHistory(JSON.parse(saved));
        loadSample();
    }, []);

    const loadSample = async () => {
        try {
            const res = await getSamplePatient();
            setSample(res.data);
        } catch { /* silent */ }
    };

    const fillSample = () => {
        if (!sample) return;
        setAge(sample.age);
        setGender(sample.gender);
        setSmoking(sample.smoking_history);
        setStage(sample.cancer_stage);
        setGeneValues(sample.gene_values);
        setMethylValues(sample.methylation_values);
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const payload: PatientData = {
                age, gender, smoking_history: smoking, cancer_stage: stage,
                gene_values: geneValues, methylation_values: methylValues,
            };
            const res = await predict(payload);
            setResult(res.data);
            const entry = { date: new Date().toLocaleString(), prob: res.data.survival_probability, risk: res.data.risk_level };
            const newHistory = [entry, ...history].slice(0, 20);
            setHistory(newHistory);
            localStorage.setItem('gxe_history', JSON.stringify(newHistory));
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const geneNames = sample?.gene_names || [];
    const cpgNames = sample?.cpg_names || [];

    return (
        <div className="max-w-7xl mx-auto px-6 py-12">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <h1 className="text-3xl font-bold mb-2">Survival Prediction</h1>
                <p className="text-slate-400 mb-8">Enter patient data to predict lung cancer survival probability.</p>
            </motion.div>

            <div className="grid lg:grid-cols-5 gap-8">
                {/* Input Form */}
                <motion.div className="lg:col-span-3 space-y-6" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
                    {/* Clinical */}
                    <div className="bg-gradient-card rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-5">
                            <h2 className="text-lg font-semibold">Clinical Information</h2>
                            <button onClick={fillSample} className="text-xs px-3 py-1.5 rounded-lg bg-blue-600/20 text-blue-400 border border-blue-500/30 hover:bg-blue-600/30 transition">
                                📋 Sample Patient
                            </button>
                        </div>

                        <div className="grid sm:grid-cols-2 gap-4">
                            <div>
                                <label className="text-xs text-slate-500 mb-1 block">Age: {age}</label>
                                <input type="range" min={20} max={95} value={age} onChange={e => setAge(+e.target.value)}
                                    className="w-full accent-blue-500" />
                            </div>
                            <div>
                                <label className="text-xs text-slate-500 mb-1 block">Gender</label>
                                <select value={gender} onChange={e => setGender(e.target.value)}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500 outline-none">
                                    <option value="MALE">Male</option>
                                    <option value="FEMALE">Female</option>
                                </select>
                            </div>
                            <div>
                                <label className="text-xs text-slate-500 mb-1 block">Smoking History</label>
                                <select value={smoking} onChange={e => setSmoking(e.target.value)}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500 outline-none">
                                    <option value="never">Never Smoker</option>
                                    <option value="former_gt15">Former (&gt;15 yrs)</option>
                                    <option value="former_le15">Former (≤15 yrs)</option>
                                    <option value="current">Current Smoker</option>
                                </select>
                            </div>
                            <div>
                                <label className="text-xs text-slate-500 mb-1 block">Cancer Stage</label>
                                <select value={stage} onChange={e => setStage(e.target.value)}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500 outline-none">
                                    <option value="I">Stage I</option>
                                    <option value="II">Stage II</option>
                                    <option value="III">Stage III</option>
                                    <option value="IV">Stage IV</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Gene Expression */}
                    <div className="bg-gradient-card rounded-2xl p-6">
                        <h2 className="text-lg font-semibold mb-4">🧬 Gene Expression (Top 10)</h2>
                        <div className="grid sm:grid-cols-2 gap-3">
                            {geneNames.slice(0, 10).map(g => (
                                <div key={g} className="flex items-center gap-2">
                                    <label className="text-xs text-slate-400 w-20 truncate" title={g}>{g}</label>
                                    <input type="number" step="0.1" value={geneValues[g] ?? ''} placeholder="0.0"
                                        onChange={e => setGeneValues(v => ({ ...v, [g]: parseFloat(e.target.value) || 0 }))}
                                        className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:border-blue-500 outline-none" />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Methylation */}
                    <div className="bg-gradient-card rounded-2xl p-6">
                        <h2 className="text-lg font-semibold mb-4">🔬 Methylation Sites (Top 5)</h2>
                        <div className="grid sm:grid-cols-2 gap-3">
                            {cpgNames.slice(0, 5).map(c => (
                                <div key={c} className="flex items-center gap-2">
                                    <label className="text-xs text-slate-400 w-24 truncate" title={c}>{c}</label>
                                    <input type="number" step="0.01" min="0" max="1" value={methylValues[c] ?? ''} placeholder="0.0-1.0"
                                        onChange={e => setMethylValues(v => ({ ...v, [c]: parseFloat(e.target.value) || 0 }))}
                                        className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:border-blue-500 outline-none" />
                                </div>
                            ))}
                        </div>
                    </div>

                    <button onClick={handleSubmit} disabled={loading}
                        className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-semibold shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 disabled:opacity-50 transition-all text-lg">
                        {loading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" /></svg>
                                Predicting...
                            </span>
                        ) : 'Predict Survival →'}
                    </button>

                    {error && <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">{error}</div>}
                </motion.div>

                {/* Results Panel */}
                <motion.div className="lg:col-span-2 space-y-6" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
                    <AnimatePresence mode="wait">
                        {result ? (
                            <motion.div key="result" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                                {/* Gauge */}
                                <div className="bg-gradient-card rounded-2xl p-6 mb-6 glow-blue">
                                    <h3 className="text-sm text-slate-500 text-center mb-4">Survival Prediction</h3>
                                    <GaugeChart value={result.survival_probability} />
                                    <div className="text-center mt-4">
                                        <span className={`inline-block px-4 py-1.5 rounded-full text-sm font-semibold ${result.risk_level === 'Low' ? 'bg-green-500/15 text-green-400 border border-green-500/30' :
                                                result.risk_level === 'Medium' ? 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/30' :
                                                    'bg-red-500/15 text-red-400 border border-red-500/30'
                                            }`}>
                                            {result.risk_level} Risk
                                        </span>
                                    </div>
                                    <div className="mt-4 flex justify-center">
                                        <div className="text-center">
                                            <div className="text-xs text-slate-500">Confidence</div>
                                            <div className="text-lg font-semibold text-blue-400">{(result.confidence * 100).toFixed(0)}%</div>
                                        </div>
                                    </div>
                                </div>

                                {/* Feature Contributions */}
                                <div className="bg-gradient-card rounded-2xl p-6">
                                    <h3 className="text-sm font-semibold mb-4 text-slate-300">Feature Contributions</h3>
                                    <div className="text-xs text-slate-500 flex justify-between mb-2 px-36">
                                        <span>← Lower Risk</span><span>Higher Risk →</span>
                                    </div>
                                    {result.contributions.slice(0, 10).map(c => (
                                        <ContributionBar key={c.feature} feature={c.feature} contribution={c.contribution} />
                                    ))}
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div key="placeholder" className="bg-gradient-card rounded-2xl p-12 text-center">
                                <div className="text-5xl mb-4">🔬</div>
                                <p className="text-slate-500">Enter patient data and click<br /><strong className="text-slate-400">Predict Survival</strong> to see results</p>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* History */}
                    {history.length > 0 && (
                        <div className="bg-gradient-card rounded-2xl p-6">
                            <h3 className="text-sm font-semibold mb-3 text-slate-300">Recent Predictions</h3>
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                                {history.slice(0, 5).map((h, i) => (
                                    <div key={i} className="flex items-center justify-between text-xs py-1.5 border-b border-slate-800 last:border-0">
                                        <span className="text-slate-500">{h.date}</span>
                                        <span className={`font-semibold ${h.risk === 'Low' ? 'text-green-400' : h.risk === 'Medium' ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {(h.prob * 100).toFixed(1)}% — {h.risk}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    );
}
