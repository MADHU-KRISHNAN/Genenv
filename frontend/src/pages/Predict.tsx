import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    predict, predictFile, getSamplePatient, getTemplate,
    type PatientData, type PredictionResult, type SamplePatient, type TemplateData
} from '../api/client';

/* ───────── Gauge Chart ───────── */
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

/* ───────── Contribution Bar ───────── */
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

/* ───────── Feature Coverage Badge ───────── */
function CoverageBadge({ provided, total }: { provided: number; total: number }) {
    const pct = Math.round((provided / total) * 100);
    const color = pct > 80 ? 'text-green-400 border-green-500/30 bg-green-500/10'
        : pct > 20 ? 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10'
            : 'text-red-400 border-red-500/30 bg-red-500/10';
    return (
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${color}`}>
            📊 {provided}/{total} features ({pct}%)
        </span>
    );
}

/* ═══════════════════════════════════════════ */
/*                 MAIN PAGE                   */
/* ═══════════════════════════════════════════ */
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
    const [history, setHistory] = useState<{ date: string; prob: number; risk: string; mode: string }[]>([]);

    // Input mode: "manual" or "file"
    const [inputMode, setInputMode] = useState<'manual' | 'file'>('manual');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

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
            let res;
            if (inputMode === 'file' && selectedFile) {
                res = await predictFile(selectedFile);
            } else {
                const payload: PatientData = {
                    age, gender, smoking_history: smoking, cancer_stage: stage,
                    gene_values: geneValues, methylation_values: methylValues,
                };
                res = await predict(payload);
            }
            setResult(res.data);
            const entry = {
                date: new Date().toLocaleString(),
                prob: res.data.survival_probability,
                risk: res.data.risk_level,
                mode: res.data.input_mode,
            };
            const newHistory = [entry, ...history].slice(0, 20);
            setHistory(newHistory);
            localStorage.setItem('gxe_history', JSON.stringify(newHistory));
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const downloadTemplate = async () => {
        try {
            const res = await getTemplate();
            const tmpl: TemplateData = res.data;
            const csvContent = tmpl.columns.join(',') + '\n' + tmpl.example_row.join(',') + '\n';
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'gxe_patient_template.csv';
            a.click();
            URL.revokeObjectURL(url);
        } catch {
            setError('Failed to download template.');
        }
    };

    const handleFileDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.csv') || file.name.endsWith('.tsv') || file.name.endsWith('.txt'))) {
            setSelectedFile(file);
        } else {
            setError('Please upload a .csv or .tsv file');
        }
    };

    const geneNames = sample?.gene_names || [];
    const cpgNames = sample?.cpg_names || [];

    return (
        <div className="max-w-7xl mx-auto px-6 py-12">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <h1 className="text-3xl font-bold mb-2">Survival Prediction</h1>
                <p className="text-slate-400 mb-6">Enter patient data to predict lung cancer survival probability.</p>

                {/* ── Imputation Notice ── */}
                <div className="mb-8 p-4 rounded-xl bg-blue-500/5 border border-blue-500/20">
                    <div className="flex items-start gap-3">
                        <span className="text-blue-400 text-lg mt-0.5">ℹ️</span>
                        <div className="text-sm text-slate-400">
                            <span className="text-blue-400 font-semibold">Median Imputation Active:</span> Features you don't provide
                            are automatically filled with training-set median values (not zeros), ensuring realistic predictions
                            even with partial input. For best accuracy, use <strong className="text-slate-300">File Upload</strong> with
                            full genomic data.
                        </div>
                    </div>
                </div>
            </motion.div>

            <div className="grid lg:grid-cols-5 gap-8">
                {/* ════════════ Input Form ════════════ */}
                <motion.div className="lg:col-span-3 space-y-6" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>

                    {/* ── Mode Toggle ── */}
                    <div className="flex rounded-xl overflow-hidden border border-slate-700/50">
                        <button
                            onClick={() => setInputMode('manual')}
                            className={`flex-1 py-3 text-sm font-semibold transition-all ${inputMode === 'manual'
                                ? 'bg-gradient-to-r from-blue-600/30 to-cyan-600/20 text-blue-400 border-b-2 border-blue-500'
                                : 'bg-slate-800/50 text-slate-500 hover:text-slate-300'
                                }`}
                        >
                            ✏️ Manual Entry
                        </button>
                        <button
                            onClick={() => setInputMode('file')}
                            className={`flex-1 py-3 text-sm font-semibold transition-all ${inputMode === 'file'
                                ? 'bg-gradient-to-r from-purple-600/30 to-pink-600/20 text-purple-400 border-b-2 border-purple-500'
                                : 'bg-slate-800/50 text-slate-500 hover:text-slate-300'
                                }`}
                        >
                            📁 File Upload (Full Data)
                        </button>
                    </div>

                    <AnimatePresence mode="wait">
                        {inputMode === 'manual' ? (
                            <motion.div key="manual" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
                                {/* ── Clinical ── */}
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

                                {/* ── Gene Expression (Top 20) ── */}
                                <div className="bg-gradient-card rounded-2xl p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <h2 className="text-lg font-semibold">🧬 Gene Expression (Top {geneNames.length})</h2>
                                        <span className="text-xs text-slate-600">
                                            Remaining {2000 - geneNames.length} genes → median imputed
                                        </span>
                                    </div>
                                    <div className="grid sm:grid-cols-2 gap-3">
                                        {geneNames.map(g => (
                                            <div key={g} className="flex items-center gap-2">
                                                <label className="text-xs text-slate-400 w-20 truncate" title={g}>{g}</label>
                                                <input type="number" step="0.1" value={geneValues[g] ?? ''} placeholder="0.0"
                                                    onChange={e => setGeneValues(v => ({ ...v, [g]: parseFloat(e.target.value) || 0 }))}
                                                    className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:border-blue-500 outline-none" />
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* ── Methylation (Top 10) ── */}
                                <div className="bg-gradient-card rounded-2xl p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <h2 className="text-lg font-semibold">🔬 Methylation Sites (Top {cpgNames.length})</h2>
                                        <span className="text-xs text-slate-600">
                                            Remaining {5000 - cpgNames.length} CpGs → median imputed
                                        </span>
                                    </div>
                                    <div className="grid sm:grid-cols-2 gap-3">
                                        {cpgNames.map(c => (
                                            <div key={c} className="flex items-center gap-2">
                                                <label className="text-xs text-slate-400 w-24 truncate" title={c}>{c}</label>
                                                <input type="number" step="0.01" min="0" max="1" value={methylValues[c] ?? ''} placeholder="0.0-1.0"
                                                    onChange={e => setMethylValues(v => ({ ...v, [c]: parseFloat(e.target.value) || 0 }))}
                                                    className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:border-blue-500 outline-none" />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </motion.div>
                        ) : (
                            /* ════════════ FILE UPLOAD MODE ════════════ */
                            <motion.div key="file" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
                                <div className="bg-gradient-card rounded-2xl p-6">
                                    <h2 className="text-lg font-semibold mb-2">📁 Upload Patient Data File</h2>
                                    <p className="text-sm text-slate-500 mb-5">
                                        Upload a CSV or TSV file with all 7,004 features for maximum prediction accuracy.
                                        Missing columns will be filled with training-set medians.
                                    </p>

                                    {/* Drop Zone */}
                                    <div
                                        className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer
                                            ${dragOver
                                                ? 'border-purple-400 bg-purple-500/10'
                                                : selectedFile
                                                    ? 'border-green-500/50 bg-green-500/5'
                                                    : 'border-slate-700 hover:border-slate-500 bg-slate-800/30'}`}
                                        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                                        onDragLeave={() => setDragOver(false)}
                                        onDrop={handleFileDrop}
                                        onClick={() => fileInputRef.current?.click()}
                                    >
                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            accept=".csv,.tsv,.txt"
                                            className="hidden"
                                            onChange={e => {
                                                const f = e.target.files?.[0];
                                                if (f) setSelectedFile(f);
                                            }}
                                        />
                                        {selectedFile ? (
                                            <div>
                                                <div className="text-4xl mb-3">✅</div>
                                                <p className="text-green-400 font-semibold">{selectedFile.name}</p>
                                                <p className="text-xs text-slate-500 mt-1">
                                                    {(selectedFile.size / 1024).toFixed(1)} KB — Click or drop to replace
                                                </p>
                                            </div>
                                        ) : (
                                            <div>
                                                <div className="text-4xl mb-3">📄</div>
                                                <p className="text-slate-400 font-medium">
                                                    Drag & drop your CSV/TSV file here
                                                </p>
                                                <p className="text-xs text-slate-600 mt-1">or click to browse</p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Template Download */}
                                    <div className="mt-5 flex items-center justify-between p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                                        <div>
                                            <p className="text-sm text-slate-300 font-medium">Need a template?</p>
                                            <p className="text-xs text-slate-500">Download a CSV with all 7,004 feature columns pre-filled with example values</p>
                                        </div>
                                        <button
                                            onClick={downloadTemplate}
                                            className="px-4 py-2 rounded-lg bg-purple-600/20 text-purple-400 border border-purple-500/30 hover:bg-purple-600/30 transition text-sm font-medium whitespace-nowrap"
                                        >
                                            ⬇️ Template CSV
                                        </button>
                                    </div>

                                    {/* File Format Info */}
                                    <div className="mt-5 p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                                        <p className="text-xs text-slate-500 font-semibold mb-2">FILE FORMAT:</p>
                                        <ul className="text-xs text-slate-600 space-y-1 list-disc list-inside">
                                            <li>Header row with column names (gene names, CpG sites, clinical fields)</li>
                                            <li>One data row per patient</li>
                                            <li>Required: <code className="text-slate-400">age, gender, smoking_history, cancer_stage</code></li>
                                            <li>Gene values = raw expression counts (will be log-transformed)</li>
                                            <li>Methylation values = beta values (0 to 1)</li>
                                            <li>You can include <strong className="text-slate-400">any subset</strong> of the 7,004 features — rest are auto-filled</li>
                                        </ul>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* ── Submit Button ── */}
                    <button
                        onClick={handleSubmit}
                        disabled={loading || (inputMode === 'file' && !selectedFile)}
                        className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-semibold shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 disabled:opacity-50 transition-all text-lg"
                    >
                        {loading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" /></svg>
                                Predicting...
                            </span>
                        ) : inputMode === 'file' ? 'Upload & Predict →' : 'Predict Survival →'}
                    </button>

                    {error && <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">{error}</div>}
                </motion.div>

                {/* ════════════ Results Panel ════════════ */}
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
                                    <div className="mt-4 flex justify-center gap-6">
                                        <div className="text-center">
                                            <div className="text-xs text-slate-500">Confidence</div>
                                            <div className="text-lg font-semibold text-blue-400">{(result.confidence * 100).toFixed(0)}%</div>
                                        </div>
                                        <div className="text-center">
                                            <div className="text-xs text-slate-500">Input Mode</div>
                                            <div className="text-sm font-semibold text-purple-400 capitalize">{result.input_mode}</div>
                                        </div>
                                    </div>
                                    <div className="mt-3 flex justify-center">
                                        <CoverageBadge provided={result.features_provided} total={result.features_total} />
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
                                <div className="mt-6 p-4 rounded-xl bg-slate-800/40 text-left">
                                    <p className="text-xs text-slate-500 font-semibold mb-2">💡 TIP</p>
                                    <p className="text-xs text-slate-600">
                                        For quick testing, click <strong className="text-slate-400">"Sample Patient"</strong> to auto-fill the form.
                                        For best accuracy, use <strong className="text-slate-400">"File Upload"</strong> with full genomic data.
                                    </p>
                                </div>
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
                                        <div className="flex items-center gap-2">
                                            <span className={`w-1.5 h-1.5 rounded-full ${h.mode === 'file' ? 'bg-purple-400' : 'bg-blue-400'}`} />
                                            <span className="text-slate-500">{h.date}</span>
                                        </div>
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
