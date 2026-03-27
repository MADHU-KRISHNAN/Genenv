import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const navItems = [
    { path: '/', label: 'Home' },
    { path: '/predict', label: 'Predict' },
    { path: '/insights', label: 'Model Insights' },
    { path: '/about', label: 'About' },
];

export default function Navbar() {
    const location = useLocation();

    return (
        <nav className="sticky top-0 z-50 border-b border-blue-900/30 bg-[#0a0f1e]/80 backdrop-blur-xl">
            <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
                <Link to="/" className="flex items-center gap-3 group">
                    <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/25">
                        GxE
                    </div>
                    <span className="text-lg font-semibold text-white group-hover:text-blue-400 transition-colors">
                        GxE Predictor
                    </span>
                </Link>

                <div className="flex items-center gap-1">
                    {navItems.map((item) => {
                        const active = location.pathname === item.path;
                        return (
                            <Link key={item.path} to={item.path} className="relative px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                                {active && (
                                    <motion.div
                                        layoutId="nav-indicator"
                                        className="absolute inset-0 bg-blue-600/20 border border-blue-500/30 rounded-lg"
                                        transition={{ type: 'spring', duration: 0.5 }}
                                    />
                                )}
                                <span className={`relative z-10 ${active ? 'text-blue-400' : 'text-slate-400 hover:text-white'}`}>
                                    {item.label}
                                </span>
                            </Link>
                        );
                    })}
                </div>
            </div>
        </nav>
    );
}
