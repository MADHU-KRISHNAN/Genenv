import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Predict from './pages/Predict';
import Insights from './pages/Insights';
import About from './pages/About';

export default function App() {
    return (
        <BrowserRouter>
            <div className="min-h-screen bg-gradient-hero">
                <Navbar />
                <main>
                    <Routes>
                        <Route path="/" element={<Landing />} />
                        <Route path="/predict" element={<Predict />} />
                        <Route path="/insights" element={<Insights />} />
                        <Route path="/about" element={<About />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    );
}
