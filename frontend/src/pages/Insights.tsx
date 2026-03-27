import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { getMetrics, getFeatureImportance, getTrainingHistory, type ModelMetrics, type FeatureImportance, type TrainingHistory } from '../api/client';

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

function MetricCard({ label, value, icon }: { label: string; value: string; icon: string }) {
    return (
        <motion.div className="bg-gradient-card rounded-2xl p-5 card-hover" whileHover={{ scale: 1.02 }}>
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
                    <p className="text-3xl font-bold mt-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">{value}</p>
                </div>
                <span className="text-2xl">{icon}</span>
            </div>
        </motion.div>
    );
}

export default function Insights() {
    const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
    const [fi, setFI] = useState<FeatureImportance | null>(null);
    const [history, setHistory] = useState<TrainingHistory | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const load = async () => {
            try {
                const [m, f, h] = await Promise.all([getMetrics(), getFeatureImportance(), getTrainingHistory()]);
                setMetrics(m.data);
                setFI(f.data);
                setHistory(h.data);
            } catch (err: any) {
                setError('Could not load model data. Is the backend running?');
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    if (loading) return (
        <div className="flex items-center justify-center min-h-[60vh]">
            <div className="animate-spin h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full" />
        </div>
    );

    if (error) return (
        <div className="max-w-2xl mx-auto px-6 py-20 text-center">
            <p className="text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl p-6">{error}</p>
        </div>
    );

    const lossData = history ? history.train_loss.map((_, i) => ({
        epoch: i + 1, train_loss: history.train_loss[i], val_loss: history.val_loss[i],
    })) : [];

    const accData = history ? history.train_acc.map((_, i) => ({
        epoch: i + 1, train_acc: +(history.train_acc[i] * 100).toFixed(1), val_acc: +(history.val_acc[i] * 100).toFixed(1),
    })) : [];

    const fiData = fi ? fi.top_features.map(f => ({
        name: f.feature.length > 12 ? f.feature.slice(0, 12) + '…' : f.feature,
        importance: +(f.importance * 1000).toFixed(2),
        category: f.category,
    })) : [];

    const pieData = fi ? [
        { name: 'Gene', value: fi.category_contributions.gene },
        { name: 'Environment', value: fi.category_contributions.environment },
        { name: 'Methylation', value: fi.category_contributions.methylation },
    ] : [];

    return (
        <div className="max-w-7xl mx-auto px-6 py-12">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <h1 className="text-3xl font-bold mb-2">Model Insights</h1>
                <p className="text-slate-400 mb-8">Training metrics, performance evaluation, and feature importance analysis.</p>
            </motion.div>

            {/* Metrics Cards */}
            {metrics && (
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
                    <MetricCard label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`} icon="🎯" />
                    <MetricCard label="AUC-ROC" value={metrics.auc_roc.toFixed(3)} icon="📈" />
                    <MetricCard label="F1 Score" value={metrics.f1_score.toFixed(3)} icon="⚖️" />
                    <MetricCard label="C-Index" value={metrics.c_index.toFixed(3)} icon="📊" />
                </div>
            )}

            {/* Charts Grid */}
            <div className="grid lg:grid-cols-2 gap-6 mb-10">
                {/* Loss Curves */}
                <motion.div className="bg-gradient-card rounded-2xl p-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                    <h3 className="text-sm font-semibold text-slate-300 mb-4">Training & Validation Loss</h3>
                    <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={lossData}>
                            <defs>
                                <linearGradient id="gradBlue" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="gradCyan" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 11 }} />
                            <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} />
                            <Legend wrapperStyle={{ fontSize: 12 }} />
                            <Area type="monotone" dataKey="train_loss" stroke="#3b82f6" fill="url(#gradBlue)" name="Train Loss" />
                            <Area type="monotone" dataKey="val_loss" stroke="#06b6d4" fill="url(#gradCyan)" name="Val Loss" />
                        </AreaChart>
                    </ResponsiveContainer>
                </motion.div>

                {/* Accuracy Curves */}
                <motion.div className="bg-gradient-card rounded-2xl p-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                    <h3 className="text-sm font-semibold text-slate-300 mb-4">Training & Validation Accuracy</h3>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={accData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 11 }} />
                            <YAxis stroke="#475569" tick={{ fontSize: 11 }} unit="%" />
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} />
                            <Legend wrapperStyle={{ fontSize: 12 }} />
                            <Line type="monotone" dataKey="train_acc" stroke="#22c55e" strokeWidth={2} dot={false} name="Train Acc" />
                            <Line type="monotone" dataKey="val_acc" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Acc" />
                        </LineChart>
                    </ResponsiveContainer>
                </motion.div>

                {/* Feature Importance */}
                <motion.div className="bg-gradient-card rounded-2xl p-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
                    <h3 className="text-sm font-semibold text-slate-300 mb-4">Top 20 Feature Importance</h3>
                    <ResponsiveContainer width="100%" height={380}>
                        <BarChart data={fiData} layout="vertical" margin={{ left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis type="number" stroke="#475569" tick={{ fontSize: 10 }} />
                            <YAxis dataKey="name" type="category" width={90} stroke="#475569" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }} />
                            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                {fiData.map((entry, i) => (
                                    <Cell key={i} fill={entry.category === 'gene' ? '#3b82f6' : entry.category === 'environment' ? '#22c55e' : '#f59e0b'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div className="flex gap-4 mt-2 justify-center text-xs text-slate-500">
                        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500" /> Gene</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500" /> Environment</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-yellow-500" /> Methylation</span>
                    </div>
                </motion.div>

                {/* Category Pie Chart */}
                <motion.div className="bg-gradient-card rounded-2xl p-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                    <h3 className="text-sm font-semibold text-slate-300 mb-4">Gene vs Environment vs Methylation Contribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie data={pieData} cx="50%" cy="50%" innerRadius={70} outerRadius={110}
                                paddingAngle={4} dataKey="value" label={({ name, value }) => `${name}: ${value}%`}
                                labelLine={{ stroke: '#475569' }}>
                                {pieData.map((_, i) => (
                                    <Cell key={i} fill={COLORS[i]} stroke="none" />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </motion.div>
            </div>
        </div>
    );
}
