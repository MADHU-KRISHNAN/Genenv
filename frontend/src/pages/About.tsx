import { motion } from 'framer-motion';

export default function About() {
    return (
        <div className="max-w-5xl mx-auto px-6 py-12">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <h1 className="text-3xl font-bold mb-2">About GxE Predictor</h1>
                <p className="text-slate-400 mb-10">Understanding gene-environment interactions in lung cancer.</p>
            </motion.div>

            {/* GxE Concept */}
            <motion.section className="bg-gradient-card rounded-2xl p-8 mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">🧬 What is Gene-Environment Interaction?</h2>
                <div className="text-slate-400 space-y-3 leading-relaxed">
                    <p>
                        Gene-Environment Interaction (GxE) refers to the phenomenon where the effect of a genetic variant on disease risk
                        depends on environmental exposures, and vice versa. In lung cancer, this means that a patient's genetic profile
                        (gene expression, DNA methylation) interacts with lifestyle factors (smoking history, age) to influence survival outcomes.
                    </p>
                    <p>
                        Our model captures these complex interactions by using a <strong className="text-white">multi-branch neural network</strong> that
                        processes genetic, epigenetic, and clinical data through separate pathways before fusing them together, allowing the model
                        to learn non-linear interaction effects that traditional statistical models might miss.
                    </p>
                </div>
            </motion.section>

            {/* Architecture */}
            <motion.section className="bg-gradient-card rounded-2xl p-8 mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">🏗️ Neural Network Architecture</h2>
                <div className="overflow-x-auto">
                    <div className="min-w-[600px] flex items-center justify-center gap-4 py-4">
                        {/* Gene Branch */}
                        <div className="flex flex-col items-center gap-2">
                            <div className="px-4 py-2 rounded-lg bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs font-mono text-center">
                                Gene Expression<br />(2,000 features)
                            </div>
                            <div className="w-px h-4 bg-blue-500/30" />
                            <div className="px-3 py-1.5 rounded bg-blue-500/10 text-xs text-blue-300">BatchNorm → 512 → 256</div>
                        </div>

                        {/* Arrow to interaction */}
                        <div className="flex flex-col items-center gap-2 mt-8">
                            <div className="text-slate-600 text-lg">→</div>
                        </div>

                        {/* Interaction Layer */}
                        <div className="flex flex-col items-center gap-2">
                            <div className="px-4 py-3 rounded-lg bg-purple-600/20 border border-purple-500/30 text-purple-400 text-xs font-mono text-center">
                                Interaction Layer<br />(Gene × Env)
                            </div>
                            <div className="w-px h-4 bg-purple-500/30" />
                            <div className="px-3 py-1.5 rounded bg-purple-500/10 text-xs text-purple-300">320 → 256 → 128</div>
                        </div>

                        <div className="flex flex-col items-center gap-2 mt-8">
                            <div className="text-slate-600 text-lg">→</div>
                        </div>

                        {/* Fusion */}
                        <div className="flex flex-col items-center gap-2">
                            <div className="px-4 py-3 rounded-lg bg-cyan-600/20 border border-cyan-500/30 text-cyan-400 text-xs font-mono text-center">
                                Final Fusion<br />(All branches)
                            </div>
                            <div className="w-px h-4 bg-cyan-500/30" />
                            <div className="px-3 py-1.5 rounded bg-cyan-500/10 text-xs text-cyan-300">256 → 64 → 1 (Sigmoid)</div>
                        </div>
                    </div>

                    <div className="min-w-[600px] flex items-start justify-center gap-4">
                        {/* Env Branch */}
                        <div className="flex flex-col items-center gap-2">
                            <div className="px-4 py-2 rounded-lg bg-green-600/20 border border-green-500/30 text-green-400 text-xs font-mono text-center">
                                Clinical / Env<br />(4 features)
                            </div>
                            <div className="w-px h-4 bg-green-500/30" />
                            <div className="px-3 py-1.5 rounded bg-green-500/10 text-xs text-green-300">BatchNorm → 128 → 64</div>
                            <div className="text-lg text-slate-600">↗</div>
                        </div>

                        <div className="w-20" />

                        {/* Methylation Branch */}
                        <div className="flex flex-col items-center gap-2">
                            <div className="px-4 py-2 rounded-lg bg-yellow-600/20 border border-yellow-500/30 text-yellow-400 text-xs font-mono text-center">
                                DNA Methylation<br />(5,000 CpG sites)
                            </div>
                            <div className="w-px h-4 bg-yellow-500/30" />
                            <div className="px-3 py-1.5 rounded bg-yellow-500/10 text-xs text-yellow-300">256 → 128</div>
                            <div className="text-lg text-slate-600">↗</div>
                        </div>
                    </div>
                </div>
            </motion.section>

            {/* Dataset */}
            <motion.section className="bg-gradient-card rounded-2xl p-8 mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">📁 Dataset Description</h2>
                <div className="text-slate-400 space-y-3 leading-relaxed">
                    <p>
                        This application uses data from the <strong className="text-white">TCGA Lung Adenocarcinoma (LUAD)</strong> cohort,
                        accessed via the UCSC Xena Browser. The dataset contains multi-omics data from 454 patients.
                    </p>
                    <div className="grid sm:grid-cols-3 gap-4 mt-4">
                        <div className="p-4 rounded-xl bg-blue-600/10 border border-blue-500/20">
                            <h4 className="font-semibold text-white text-sm mb-1">Gene Expression</h4>
                            <p className="text-xs text-slate-500">IlluminaHiSeq RNAseq, 20,530 genes → top 2,000 by variance</p>
                        </div>
                        <div className="p-4 rounded-xl bg-green-600/10 border border-green-500/20">
                            <h4 className="font-semibold text-white text-sm mb-1">Clinical Data</h4>
                            <p className="text-xs text-slate-500">Age, gender, smoking history, cancer stage, vital status, survival time</p>
                        </div>
                        <div className="p-4 rounded-xl bg-yellow-600/10 border border-yellow-500/20">
                            <h4 className="font-semibold text-white text-sm mb-1">DNA Methylation</h4>
                            <p className="text-xs text-slate-500">HumanMethylation450, 485K CpG sites → top 5,000 by variance</p>
                        </div>
                    </div>
                </div>
            </motion.section>

            {/* Citations */}
            <motion.section className="bg-gradient-card rounded-2xl p-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">📚 Citations</h2>
                <div className="text-slate-400 text-sm space-y-2">
                    <p>• The Cancer Genome Atlas Research Network. <em>Comprehensive molecular profiling of lung adenocarcinoma.</em> Nature 511, 543–550 (2014).</p>
                    <p>• Goldman, M.J. et al. <em>Visualizing and interpreting cancer genomics data via the Xena platform.</em> Nat. Biotechnol. 38, 675–678 (2020).</p>
                    <p>• Lundberg, S.M. & Lee, S. <em>A Unified Approach to Interpreting Model Predictions.</em> NeurIPS 2017.</p>
                </div>
            </motion.section>
        </div>
    );
}
