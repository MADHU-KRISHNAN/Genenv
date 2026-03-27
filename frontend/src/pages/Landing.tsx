import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const fadeUp = {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 },
};

const features = [
    { icon: '🧬', title: 'Multi-Omics Integration', desc: 'Combines gene expression, DNA methylation, and clinical data for comprehensive analysis.' },
    { icon: '🤖', title: 'Deep Learning Model', desc: 'Dual-branch neural network captures gene-environment interactions for survival prediction.' },
    { icon: '📊', title: 'Explainable AI', desc: 'SHAP-based feature importance reveals which factors drive each prediction.' },
    { icon: '🎯', title: 'Personalized Risk', desc: 'Patient-specific survival probability with confidence intervals and risk stratification.' },
];

export default function Landing() {
    return (
        <div className="overflow-hidden">
            {/* Hero Section */}
            <section className="relative min-h-[85vh] flex items-center justify-center px-6">
                {/* Animated background orbs */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-blue-600/10 blur-3xl animate-pulse" />
                    <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
                    <div className="absolute top-1/2 left-1/2 w-64 h-64 rounded-full bg-purple-500/5 blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
                </div>

                <div className="relative max-w-4xl mx-auto text-center">
                    <motion.div {...fadeUp}>
                        <span className="inline-block px-4 py-1.5 mb-6 rounded-full text-xs font-semibold tracking-wider uppercase bg-blue-600/15 text-blue-400 border border-blue-500/20">
                            TCGA-LUAD Dataset • PyTorch Deep Learning
                        </span>
                    </motion.div>

                    <motion.h1
                        className="text-5xl md:text-7xl font-extrabold leading-tight mb-6"
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.1 }}
                    >
                        <span className="bg-gradient-to-r from-white via-blue-100 to-blue-300 bg-clip-text text-transparent">
                            Gene–Environment
                        </span>
                        <br />
                        <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                            Interaction Predictor
                        </span>
                    </motion.h1>

                    <motion.p
                        className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7, delay: 0.3 }}
                    >
                        Predict lung cancer survival outcomes using a multi-branch deep learning model
                        that integrates gene expression, DNA methylation, and clinical data from the
                        TCGA Lung Adenocarcinoma cohort.
                    </motion.p>

                    <motion.div
                        className="flex flex-wrap gap-4 justify-center"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.5 }}
                    >
                        <Link
                            to="/predict"
                            className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 text-white font-semibold shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 transition-all"
                        >
                            Start Prediction →
                        </Link>
                        <Link
                            to="/insights"
                            className="px-8 py-3.5 rounded-xl bg-white/5 border border-white/10 text-white font-semibold hover:bg-white/10 transition-all"
                        >
                            View Model Insights
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Features */}
            <section className="py-24 px-6">
                <div className="max-w-6xl mx-auto">
                    <motion.h2
                        className="text-3xl font-bold text-center mb-4"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                    >
                        How It Works
                    </motion.h2>
                    <p className="text-slate-400 text-center mb-16 max-w-xl mx-auto">
                        Our model uses a dual-branch architecture to capture complex interactions
                        between genetic and environmental factors.
                    </p>

                    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {features.map((f, i) => (
                            <motion.div
                                key={f.title}
                                className="bg-gradient-card rounded-2xl p-6 card-hover"
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.1 }}
                            >
                                <div className="text-3xl mb-4">{f.icon}</div>
                                <h3 className="text-lg font-semibold mb-2 text-white">{f.title}</h3>
                                <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Stats */}
            <section className="py-16 px-6 border-t border-blue-900/20">
                <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
                    {[
                        { val: '454', label: 'Patient Samples' },
                        { val: '2,000', label: 'Gene Features' },
                        { val: '5,000', label: 'CpG Sites' },
                        { val: '3', label: 'Data Modalities' },
                    ].map((s) => (
                        <div key={s.label}>
                            <div className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">{s.val}</div>
                            <div className="text-sm text-slate-500 mt-1">{s.label}</div>
                        </div>
                    ))}
                </div>
            </section>
        </div>
    );
}
